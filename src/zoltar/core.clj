(ns ^{:author "Aaron Windsor",
      :doc "A simple, off-the-shelf classifier"}
  zoltar.core
  (:use zoltar.feature_testers
	zoltar.distributions))

;tester is a map of the form:
;  { :dist Distribution :testfunc TestFunction }
(defn train-tester [tester sample]
  ""
  (assoc tester :dist
    (add-point (tester :dist) ((tester :testfunc) sample))))

(defn test-tester [tester sample]
  ""
  (prob (tester :dist) ((tester :testfunc) sample)))

(defprotocol Model
  ""
  (train [this sample category] "")
  (classify [this sample] "")
  (compile-model [this] ""))

(defn annotated-max [x y]
  (if (> (last x) (last y)) x y))

(defn log2 [x]
  (let [divisor (Math/log 2)]
    (/ (Math/log x) divisor)))

; category is a vector of testers
; categories is a map from name -> category
(defrecord NaiveBayesModel [categories]
  Model
  (train [this sample category]
    (assoc this :categories
      (assoc categories category
        (vec (for [tester (get categories category (new-category))]
	       (train-tester tester sample)))))) 
  (classify [this sample]
    (first
      (reduce annotated-max
        (for [[x y] (seq categories)]
          [x (reduce + (map log2 (map test-tester y (repeat sample))))]))))
  (compile-model [this] this)) ;TODO: exclude irrelevant feature testers

(defn naive-bayes-model [] (NaiveBayesModel. {}))
