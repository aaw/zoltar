(ns ^{:author "Aaron Windsor",
      :doc "A simple, off-the-shelf classifier."}
  zoltar.boosting
  (:use [zoltar.classifiers])
  (:use [zoltar.distributions])
  (:import [zoltar.classifiers NaiveBayesModel]))

(defn generic-indicator [classifier point match-val non-match-val]
  (if (= (classify classifier (:features point)) (:category point)) match-val non-match-val))

(defn indicator [classifier point]
  (generic-indicator classifier point 1 0))

(defn err-indicator [classifier point]
  (generic-indicator classifier point 0 1))

(defn max-score [x y]
  (if (> (:score x) (:score y)) x y))

;(composed-score 1 1000) -> 829
;(composed-score 2 1000) -> 258
;(composed-score 10 1000) -> 884

;(composed-score 1 200) -> 197
;(composed-score 2 200) -> 177
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

(defn composed-classifier [classifiers alphas classes]
  (fn [sample]
    (:class (reduce max-score
      (for [c classes]
	(let [indicators (map indicator classifiers
			      (repeat {:features sample :category c}))
	      score (reduce + (map * alphas indicators))]
	  {:class c :score score}))))))

(defn parse-file-line [line]
  (let [line-seq (seq (.split line ","))]
    {:category (first line-seq)
     :features (map #(Integer/parseInt %) (rest line-seq))}))

; /home/aaron/development/machine-learning-datasets/letter-recognition.data.txt
(defn read-data [filename]
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
  (let [remove-zeros (map #(if (<= % 0.001) 0.001 %) values)
	smallest (apply min remove-zeros)
	new-values (map (comp #(Math/round %) #(/ % smallest)) remove-zeros)]
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
