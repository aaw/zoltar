(ns ^{:author "Aaron Windsor",
      :doc "A simple, off-the-shelf classifier."}
  zoltar.feature_testers
  (:use zoltar.distributions))

(defn count-occur [string item]
  (reduce + (map (fn [x] (if (= (str x) item) 1 0)) string)))

(defn make-basic-categories []
  [{:dist (exponential-sandpile-distribution)
    :testfunc (fn [x] (count x)) }
   {:dist (exponential-sandpile-distribution)
    :testfunc (fn [x] (count-occur x " ")) }
   {:dist (exponential-sandpile-distribution)
    :testfunc (fn [x] (count-occur x "-")) }])



