(ns ^{:author "Aaron Windsor",
      :doc "A simple, off-the-shelf classifier"}
  zoltar.core
  (:use zoltar.feature_testers
	zoltar.distributions
	zoltar.classifiers
	zoltar.util)
  (:import [zoltar.classifiers NaiveBayesModel])
  (:import [zoltar.classifiers BoostedBayesModel]))

(defn naive-bayes []
  (NaiveBayesModel. {} make-basic-categories))

(defn boosted-bayes [iterations]
  (BoostedBayesModel. {} make-basic-categories iterations))

(defn naive-bayes-no-features [dimension]
  (NaiveBayesModel. {} #(make-passthrough-categories dimension)))

(defn boosted-bayes-no-features [dimension iterations]
  (BoostedBayesModel. {} #(make-basic-categories dimension) iterations))

(defn k-fold-cross-validation [model data k]
  (let [partitions (partition-all k data)
	num-partitions (count partitions)
	exclude-nth (fn [s n] (flatten (concat (take n s) (drop (inc n) s))))
	score (fn [model samples] (/ (reduce + (map indicator (repeat model) samples)) (count samples)))
	avg (fn [xs] (/ (apply + xs) (count xs)))
	kth-score (fn [k] (-> model (train (exclude-nth partitions k)) (score (nth partitions k))))]
    (avg (map kth-score (range num-partitions)))))

(def datasets { :letters letters-dataset :iris iris-dataset })