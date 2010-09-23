(ns ^{:author "Aaron Windsor",
      :doc "A simple, off-the-shelf classifier"}
  zoltar.distributions)

(defprotocol Distribution
  "Represents a probability mass function"
  (prob [dist x] "Probability of equality to a particular point")
  (add-point [dist x weight] "Adds a sample point to the distribution"))

(defn inc-map [map key size]
  (assoc map key (+ size (get map key 0))))

(defn raw-prob [dist point image-size]
  (/ (get dist point 0) (max 1 image-size)))

; TODO: redo with map and reduce with merge to combine maps
(defn bump-map [dist point radius finc]
  (loop [i (- radius)
	 d dist]
    (if (<= i radius)
      (recur (inc i)
	     (inc-map d (+ point i) (finc (inc (- radius (if (>= i 0) i (- i)))))))
      d)))


; TODO: don't take floor as an argument, make floor a function of the size of the distribution
(defrecord FlooredDistribution [dist floor image-size]
  Distribution
  (prob [this x] (max floor (raw-prob dist x image-size)))
  (add-point [this x weight]
    (-> this
      (assoc :image-size (+ weight (get this :image-size 0)))
      (assoc :dist (inc-map dist x weight)))))

(defn triangle-sequence [n] (take n (iterate inc 1)))

(defrecord SandpileDistribution [dist floor radius image-size scale-point]
  Distribution
  (prob [this x] (max floor (raw-prob dist x image-size)))
  (add-point [this x weight]
    (let [finc (comp (partial * weight) scale-point)]
      (-> this
	  (assoc :image-size (+ (reduce + (map finc (triangle-sequence radius)))
				(reduce + (map finc (triangle-sequence (inc radius))))
				(get this :image-size 0)))
	  (assoc :dist (bump-map dist x radius finc))))))

(memoize (defn pow2 [x]
  (if (= x 0) 1 (* 2 (pow2 (dec x)))))) ; quick hack, will replace impl

(defn floored-distribution []
  (FlooredDistribution. {} 0.01 0))

(defn linear-sandpile-distribution []
  (SandpileDistribution. {} 0.01 2 0 identity))

(defn exponential-sandpile-distribution []
  (SandpileDistribution. {} 0.01 2 0 pow2))
