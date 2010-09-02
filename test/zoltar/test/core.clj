(ns zoltar.test.core
  (:use [zoltar.core] :reload)
  (:use [clojure.test])
  (:import [zoltar.core FlooredDistribution]))

(deftest test-floor
  (testing
    (is (= (prob (FlooredDistribution. {} 0.123) "nothing") 0.123))
    (is (= (prob (FlooredDistribution. {:a 1} 0.01) :a) 1))))
