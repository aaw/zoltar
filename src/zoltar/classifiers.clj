(ns ^{:author "Aaron Windsor",
      :doc "A simple, off-the-shelf classifier"}
  zoltar.classifiers
  (:use clojure.contrib.math
        zoltar.feature_testers
	zoltar.distributions))

;tester is a map of the form:
;  { :dist Distribution :testfunc TestFunction }
(defn train-tester [testers sample-map]
  ""
  (let [{:keys [sample category weight] :or {weight 1}} sample-map
	add-single-point (fn [tester]
			   (add-point (:dist tester) ((:testfunc tester) sample) weight))
	reconstruct-tester (fn [orig-tester new-dist]
			     {:testfunc (:testfunc orig-tester) :dist new-dist})]
    {:category category :features (map reconstruct-tester testers (map add-single-point testers))}))

(defprotocol Model
  ""
  (train [this samples]
	  "samples is a sequence of maps. Each map must have
          the keys :sample and :category, and can have a non-negative
          integer :weight. Defining a sample with weight x is equivalent
          to repeating the sample x times in the samples sequence")
  (classify [this sample]
	    "Given a sample, returns the category that best matches
             that sample based on the training data."))

(defn max-score [x y]
  (if (> (:score x) (:score y)) x y))

; category is a seq of {:category :features ({:dist :testfunc}, ...)}
(defn merge-features [features1 features2]
  "Merges two feature sets by summing their distributions. Both
   features1 and features2 are sequences of the form
   ({:dist x1 :testfunc y1}, ..., {:dist xn :testfunc yn}). It's
   assumed that these sequences have the same testfuncs in the
   same order."
  (let [merge-feature (fn [{d1 :dist func1 :testfunc}
			   {d2 :dist func2 :testfunc}] 
			{:dist (merge-dist d1 d2)
			 :testfunc func1})]
    (map merge-feature features1 features2)))

(defn combine-categories [category-seq initial-category]
  "Combines equivalent categories in a sequence. Categories
   are maps of the form:

   {:category x1 :features ({:dist y1 :testfunc z1}, ...)}

   We end up with a sequence of these maps and want to combine
   equivalent categories so that we have a sequence that contains
   exactly one map per category. For equivalent categories, we
   use merge-features to merge their feature sequences."
  (loop [cseq category-seq
	 seen-map {}]
    (if (not (seq cseq))
      (map (fn [[x y]] {:category x :features y}) (seq seen-map))
      (let [remaining (rest cseq)
	    {category :category features :features} (first cseq)
	    existing-features (get seen-map category initial-category)]
	(recur remaining (assoc seen-map category (merge-features features existing-features)))))))

; categories is a map from category name -> sequence of testers
; testers are distribution + test function
(defrecord NaiveBayesModel [categories create-category]
  Model
  (train [this samples]
    (let [initial-category (create-category)]
	 (assoc this :categories
		(combine-categories (map train-tester (repeat initial-category) samples) initial-category))))
  (classify [this sample]
    (let [test-tester (fn [tester sample]
			(prob (:dist tester) ((:testfunc tester) sample)))]
      (:category (reduce max-score
        (for [{category :category features :features} (seq categories)]
	  (let [score (reduce * (map test-tester features (repeat sample)))]
	    {:category category :score score})))))))
  
(defn generic-indicator [classifier point match-val non-match-val]
  (if (= (classify classifier (:sample point)) (:category point)) match-val non-match-val))

(defn indicator [classifier point]
  (generic-indicator classifier point 1 0))

(defn err-indicator [classifier point]
  (generic-indicator classifier point 0 1))

(defn multi-classifier-score [classifiers alphas class sample]
  (let [indicators (map indicator classifiers
			(repeat {:features sample :category class}))]
    (reduce + (map * alphas indicators))))

(defn normalize [values]
  (map / values (repeat (reduce + values))))

(defn integerize [values]
  (let [accuracy 6 ; = # decimal places accuracy
	scale (expt 10 accuracy)
	epsilon (/ 1 scale)
	remove-zeros (map #(if (<= % epsilon) epsilon %) values)
	new-values (map (comp #(round %) #(* % scale)) remove-zeros)]
    new-values))

; TODO: make this implement Model protocol
(defn composed-classifier [classifiers alphas classes]
  (fn [sample]
    (:class (reduce max-score
      (pmap (fn score [class]
	     {:class class :score (multi-classifier-score classifiers alphas class sample)}) classes)))))

; training-data is a vector of {:category :features}
(defn boost-classifier [build-classifier training-data iterations]
  (let [classes (distinct (map :category training-data))
	nclasses (count classes)
	epsilon (/ 1 (* iterations (count training-data)))
	make-weighted-sample (fn [sample weight] (assoc sample :weight weight))]
    (loop [iteration 1
	   weights (take (count training-data) (repeat 1))
	   alphas []
	   classifiers []]
      (let [classifier (build-classifier (map make-weighted-sample training-data weights))
	    classifier-indicator (map err-indicator (repeat classifier) training-data)
	    err (/ (reduce + (map * weights classifier-indicator))
		   (reduce + weights))
	    alpha (+ (Math/log (/ (- 1 err) (max err epsilon))) (Math/log (- nclasses 1)))
	    ;_ (prn classifier-indicator)
	    ;_ (prn (float err))
	    ;_ (prn alpha)
	    ;_ (prn weights)
	    new-weights ((comp integerize normalize)
			  (map * weights
			    (map #(Math/exp %)
			      (map * (repeat alpha) classifier-indicator ))))
	    ;_ (prn new-weights)
	    ;_ (prn "-------------------------------------------------")
	    new-classifiers (conj classifiers classifier)
	    new-alphas (conj alphas alpha)]
        (comment (println "Iteration " iteration ":"))
	(if (< iteration iterations)
	  (recur (inc iteration)
		 new-weights
		 new-alphas
		 new-classifiers)
	  (composed-classifier new-classifiers new-alphas classes))))))

(defrecord BoostedBayesModel [categories create-category iterations]
  Model
  (train [this samples]
	 (let [model (NaiveBayesModel. categories create-category)
	       train-model (fn [samples] (train model samples))
	       classifier (boost-classifier train-model samples iterations)]
	   (assoc this :classifier classifier)))
  (classify [this sample]
	    ((:classifier this) sample)))

