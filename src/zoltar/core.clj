(ns ^{:author "Aaron Windsor",
      :doc "A simple, off-the-shelf classifier"}
  zoltar.core
  (:use zoltar.feature_testers
	zoltar.distributions
	zoltar.classifiers)
  (:import [zoltar.classifiers NaiveBayesModel])
  (:import [zoltar.classifiers BoostedBayesModel]))

(defn naive-bayes-model []
  (NaiveBayesModel. {} make-basic-categories))

(defn boosted-bayes-model [iterations]
  (BoostedBayesModel. {} make-basic-categories iterations))