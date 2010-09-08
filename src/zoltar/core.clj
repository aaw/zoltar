(ns zoltar.core)

(defprotocol Distribution
  "Represents a probability mass function"
  (prob [dist x] "Probability of equality to a particular point")
  (add-point [dist x] "Adds a sample point to the distribution"))

(defn inc-map [map key]
  (assoc map key (+ 1 (get map key 0))))

(defrecord FlooredDistribution [dist floor]
  Distribution
  (prob [this x]
	(max floor
	     (/ (get dist x 0)
		(max 1 (reduce + (vals dist))))))
  (add-point [this x] (assoc this :dist (inc-map dist x))))

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

(defn floored-distribution []
  (FlooredDistribution. {} 0.01))

(defn count-occur [string item]
  (reduce + (map (fn [x] (if (= (str x) item) 1 0)) string)))

(defn new-category []
  [{:dist (floored-distribution)
    :testfunc (fn [x] (count x)) }
   {:dist (floored-distribution)
    :testfunc (fn [x] (count-occur x " ")) }
   {:dist (floored-distribution)
    :testfunc (fn [x] (count-occur x "-")) }])

(defn annotated-max [x y]
  (if (> (last x) (last y)) x y))

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
          [x (reduce * (map test-tester y (repeat sample)))]))))
  (compile-model [this] this)) ;TODO: exclude irrelevant feature testers

(defn naive-bayes-model [] (NaiveBayesModel. {}))
