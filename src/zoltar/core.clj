(ns ^{:author "Aaron Windsor",
      :doc "A simple, off-the-shelf classifier"}
  zoltar.core
  (:use zoltar.feature_testers
	zoltar.distributions
	zoltar.classifiers)
  (:import [zoltar.classifiers NaiveBayesModel]))

(defn naive-bayes-model [] (NaiveBayesModel. {} make-basic-categories))