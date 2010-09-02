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




