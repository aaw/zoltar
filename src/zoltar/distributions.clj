(ns ^{:author "Aaron Windsor",
      :doc "A simple, off-the-shelf classifier"}
  zoltar.distributions)

(defprotocol Distribution
  "Represents a probability mass function"
  (prob [dist x] "Probability of equality to a particular point")
  (add-point [dist x weight] "Adds a sample point to the distribution")
  (merge-dist [dist1 dist2] "Merges two distributions"))

(defn inc-map [map key size]
  (assoc map key (+ size (get map key 0))))

(defn raw-prob [dist point image-size]
  (/ (get dist point 0) (max 1 image-size)))

; todo: remove this, can do the same thing with range. don't know why this seemed like a good idea...
(defn triangle-sequence [n] (take n (iterate inc 1)))

; quick hack, will replace impl
(memoize (defn pow2 [x] (if (= x 0) 1 (* 2 (pow2 (dec x)))))) 

(defn bump-map [dist point radius finc]
  (let [r+ (inc radius)
	make-point (fn [^Integer x]
		     {(+ x point) (finc (- r+ (Math/abs x)))})]		  
    (merge-with + dist
      (apply merge (map make-point (range (- radius) r+))))))

; TODO: don't take floor as an argument, make floor a function of the size of the distribution
(defrecord FlooredDistribution [dist floor image-size]
  Distribution
  (prob [this x] (max floor (raw-prob dist x image-size)))
  (add-point [this x weight]
    (-> this
      (assoc :image-size (+ weight (get this :image-size 0)))
      (assoc :dist (inc-map dist x weight))))
  (merge-dist [this other]
    (-> this
	(assoc :image-size (+ image-size (:image-size other)))
	(assoc :dist (merge-with + dist (:dist other))))))

(defrecord SandpileDistribution [dist floor radius image-size scale-point]
  Distribution
  (prob [this x] (max floor (raw-prob dist x image-size)))
  (add-point [this x weight]
    (let [finc (comp (partial * weight) scale-point)]
      (-> this
	  (assoc :image-size (+ (reduce + (map finc (triangle-sequence radius)))
				(reduce + (map finc (triangle-sequence (inc radius))))
				(get this :image-size 0)))
	  (assoc :dist (bump-map dist x radius finc)))))
  (merge-dist [this other]
    (-> this
	(assoc :image-size (+ image-size (:image-size other)))
	(assoc :dist (merge-with + dist (:dist other))))))


(defn floored-distribution []
  (FlooredDistribution. {} 0.01 0))

(defn linear-sandpile-distribution []
  (SandpileDistribution. {} 0.01 2 0 identity))

(defn exponential-sandpile-distribution []
  (SandpileDistribution. {} 0.01 2 0 pow2))
