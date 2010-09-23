(ns ^{:author "Aaron Windsor",
      :doc "A simple, off-the-shelf classifier"}
  zoltar.classifiers
  (:use zoltar.feature_testers
	zoltar.distributions))

;tester is a map of the form:
;  { :dist Distribution :testfunc TestFunction }
(defn train-tester [tester sample weight]
  ""
  (assoc tester :dist
    (add-point (tester :dist) ((tester :testfunc) sample) weight)))

(defn test-tester [tester sample]
  ""
  (prob (tester :dist) ((tester :testfunc) sample)))

(defprotocol Model
  ""
  (train [this sample category weight] "")
  (classify [this sample] "")
  (compile-model [this] ""))

(defn max-score [x y]
  (if (> (:score x) (:score y)) x y))

; category is a vector of testers
; categories is a map from name -> category
(defrecord NaiveBayesModel [categories create-category]
  Model
  (train [this sample category weight]
    (assoc this :categories
      (assoc categories category
        (vec (for [tester (get categories category (create-category))]
	       (train-tester tester sample weight))))))
  (classify [this sample]
    (:category (reduce max-score
        (for [[x y] (seq categories)]
	  {:category x :score (reduce * (map test-tester y (repeat sample)))}))))
  (compile-model [this] this)) ;TODO: exclude irrelevant feature testers?

(defn generic-indicator [classifier point match-val non-match-val]
  (if (= (classify classifier (:features point)) (:category point)) match-val non-match-val))

(defn indicator [classifier point]
  (generic-indicator classifier point 1 0))

(defn err-indicator [classifier point]
  (generic-indicator classifier point 0 1))

(defn multi-classifier-score [classifiers alphas class sample]
  (let [indicators (map indicator classifiers
			(repeat {:features sample :category class}))]
    (reduce + (map * alphas indicators))))

; TODO: make this implement Model protocol
(defn composed-classifier [classifiers alphas classes]
  (fn [sample]
    (:class (reduce max-score
      (pmap (fn score [class]
	     {:class class :score (multi-classifier-score classifiers alphas class sample)}) classes)))))

(defn parse-file-line [^String line]
  (let [line-seq (seq (.split line ","))]
    {:category (first line-seq)
     :features (map #(Integer/parseInt %) (rest line-seq))}))

; /home/aaron/development/machine-learning-datasets/letter-recognition.data.txt
(defn read-data [^String filename]
  (with-open [reader (java.io.BufferedReader. (java.io.FileReader. filename))]
    (let [lseq (line-seq reader)]
      (vec (map parse-file-line lseq)))))

; just for testing
(def all-data (read-data "/home/aaron/development/machine-learning-datasets/letter-recognition.data.txt"))
(def tiny-data (take 100 all-data))

; 1 iteration on 100 samples -> 53
; 20 iterations on 100 samples -> 52
; 100 iterations on 100 samples -> 27
(defn boosted-bayes [all-data all-weights]
  (let [dist (floored-distribution)
	data-width (count (:features (first all-data)))
	make-category (fn [] (vec (for [i (range data-width)]
				    {:dist dist :testfunc #(nth % i)})))
	initial-model (NaiveBayesModel. {} make-category)]
    (loop [model initial-model
	   data all-data
	   weights all-weights]
      (let [{category :category sample :features} (first data)]
	(if (seq data)
	  (recur (train model sample category (first weights))
		 (rest data)
		 (rest weights))
	  model)))))

(defn normalize [values]
  (map / values (repeat (reduce + values))))

(defn integerize [values]
  (let [epsilon 0.001
	remove-zeros (map #(if (<= % epsilon) epsilon %) values)
	smallest (apply min remove-zeros)
	new-values (map (comp (fn [^Double x] (Math/round x)) #(/ % smallest)) remove-zeros)]
    new-values))

; training-data is a vector of {:category :features}
(defn boost-classifier [build-classifier training-data iterations]
  (let [classes (distinct (map :category training-data))
	nclasses (count classes)
	epsilon (/ 1 (* iterations (count training-data)))]
    (loop [iteration 1
	   weights (take (count training-data) (repeat 1))
	   alphas []
	   classifiers []]
      (let [classifier (build-classifier training-data weights)
	    classifier-indicator (map err-indicator (repeat classifier) training-data)
	    err (/ (reduce + (map * weights classifier-indicator))
		   (reduce + weights))
	    alpha (+ (Math/log (/ (- 1 err) (max err epsilon))) (Math/log (- nclasses 1)))
	    new-weights (integerize (map * weights
			  (map #(Math/exp %)
			    (map * (repeat alpha) classifier-indicator ))))
	    new-classifiers (conj classifiers classifier)
	    new-alphas (conj alphas alpha)]
        (println "Iteration " iteration ":")
	(if (< iteration iterations)
	  (recur (inc iteration)
		 new-weights
		 new-alphas
		 new-classifiers)
	  (composed-classifier new-classifiers new-alphas classes))))))

(defn composed-score [iterations num-items]
  (let [data (take num-items all-data)
	classifier (boost-classifier boosted-bayes data iterations)
	ind (fn [classy point] (if (= (classy (:features point)) (:category point)) 1 0))
	score (reduce + (map ind (repeat classifier) data))]
    score))

;(composed-score 1 1000) -> 829
;(composed-score 2 1000) -> 258
;(composed-score 10 1000) -> 884

;(composed-score 1 200) -> 197 (16 secs)
;(composed-score 2 200) -> 177 (33 secs)
;(composed-score 3 200) -> 197
;(composed-score 4 200) -> 178
;(composed-score 5 200) -> 197
;(composed-score 10 200) -> 178
;(composed-score 20 200) -> 199
;(composed-score 30 200) -> 199
;(composed-score 50 200) -> 199

;(composed-score 1 2000) -> 1593
;(composed-score 2 2000) -> 625
;(composed-score 10 2000) -> 1744

