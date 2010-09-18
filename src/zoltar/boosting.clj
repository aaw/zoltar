(ns ^{:author "Aaron Windsor",
      :doc "A simple, off-the-shelf classifier."}
  zoltar.boosting
  (:use [zoltar.classifiers])
  (:use [zoltar.distributions])
  (:import [zoltar.classifiers NaiveBayesModel]))

(defn indicator [classifier point]
  (if (= (classify classifier (:features point)) (:category point)) 1 0))

(defn max-score [x y]
  (if (> (:score x) (:score y)) x y))

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
;(def all-data (read-data "/home/aaron/development/machine-learning-datasets/letter-recognition.data.txt"))
;(def tiny-data (take 100 all-data))

(defn boosted-bayes [all-data all-weights]
  (let [dist (linear-sandpile-distribution)
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

; training-data is a vector of {:category :features}
(defn boost-classifier [build-classifier training-data iterations]
  (let [classes (distinct (map :category training-data))
	nclasses (count classes)]
    (loop [iteration 1
	   weights (normalize (take (count training-data) (repeat 1)))
	   alphas []
	   classifiers []]
      (let [classifier (build-classifier training-data weights)
	    classifier-indicator (map indicator (repeat classifier) training-data)
	    err (/ (reduce + (map * weights classifier-indicator))
		   (reduce + weights))
	    alpha (+ (Math/log (/ (- 1 err) err)) (Math/log (- nclasses 1)))
	    new-weights (normalize (map * weights
			  (map #(Math/exp %)
			    (map * (repeat alpha) classifier-indicator ))))
	    new-classifiers (conj classifiers classifier)
	    new-alphas (conj alphas alpha)]
	(print "Iteration " iteration ":")
	(print "Weights: " (take 10 weights))
	(print "Alphas: " (take 10 alphas))
	(if (< iteration iterations)
	  (recur (inc iteration)
		 new-weights
		 new-alphas
		 new-classifiers)
	  (composed-classifier new-classifiers new-alphas classes))))))
	  
	