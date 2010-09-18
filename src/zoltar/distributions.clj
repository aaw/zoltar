(ns ^{:author "Aaron Windsor",
      :doc "A simple, off-the-shelf classifier"}
  zoltar.distributions)

(defprotocol Distribution
  "Represents a probability mass function"
  (prob [dist x] "Probability of equality to a particular point")
  (add-point [dist x weight] "Adds a sample point to the distribution"))

(defn inc-map [map key size]
  (assoc map key (+ size (get map key 0))))

(defn raw-prob [dist point]
  (/ (get dist point 0)
     (max 1 (reduce + (vals dist)))))

; TODO: redo with map and reduce with merge to combine maps
(defn bump-map [dist point radius finc]
  (loop [i (- radius)
	 d dist]
    (if (<= i radius)
      (recur (inc i)
	     (inc-map d (+ point i) (finc (inc (- radius (Math/abs i))))))
      d)))

(defrecord FlooredDistribution [dist floor]
  Distribution
  (prob [this x] (max floor (raw-prob dist x)))
  (add-point [this x weight] (assoc this :dist (inc-map dist x weight))))

(defrecord LinearSandpileDistribution [dist floor radius]
  Distribution
  (prob [this x] (max floor (raw-prob dist x)))
  (add-point [this x weight]
    (assoc this :dist (bump-map dist x radius (partial * weight)))))

(memoize (defn pow2 [x]
  (if (= x 0) 1 (* 2 (pow2 (dec x)))))) ; quick hack, will replace impl

(defrecord ExponentialSandpileDistribution [dist floor radius]
  Distribution
  (prob [this x] (max floor (raw-prob dist x)))
  (add-point [this x weight]
    (assoc this :dist
      (bump-map dist x radius (comp (partial * weight) pow2)))))
	 
(defn floored-distribution []
  (FlooredDistribution. {} 0.01))

(defn linear-sandpile-distribution []
  (LinearSandpileDistribution. {} 0.01 3))

(defn exponential-sandpile-distribution []
  (ExponentialSandpileDistribution. {} 0.01 3))
