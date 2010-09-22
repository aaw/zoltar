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

(defn annotated-max [x y]
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
    (:category (reduce annotated-max
        (for [[x y] (seq categories)]
	  {:category x :score (reduce * (map test-tester y (repeat sample)))}))))
  (compile-model [this] this)) ;TODO: exclude irrelevant feature testers?
