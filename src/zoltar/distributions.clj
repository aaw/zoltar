(ns ^{:author "Aaron Windsor",
      :doc "A simple, off-the-shelf classifier"}
  zoltar.distributions)

;TODO: this is still a little gross - merge-dist shouldn't be
;      part of the interface, instead there should be a separate
;      DistributionMap protocol that takes care of dist + image-size?
(defprotocol Distribution
  "Represents a probability mass function"
  (prob [dist x] "Probability of equality to a particular point")
  (add-point [dist x weight] "Adds a sample point to the distribution")
  (merge-dist [dist1 dist2] "Merges two distributions"))

(defn inc-map [map key size]
  (assoc map key (+ size (get map key 0))))

(defn raw-prob [dist point image-size]
  (/ (get dist point 0) (max 1 image-size)))

(defn pow2 [^Integer n] (bit-shift-left 1 n))
 
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
    (let [finc (comp (partial * weight) scale-point)
	  r+ (inc radius)]
      (-> this
	  (assoc :image-size (+ (* 2 (reduce + (map finc (range 1 r+))))
				(finc r+)
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
