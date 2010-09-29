(ns zoltar.test.core
  (:use [zoltar.core] :reload)
  (:use [zoltar.classifiers] :reload)
  (:use [zoltar.distributions] :reload)
  (:use [zoltar.feature_testers] :reload)
  (:use [clojure.test])
  (:import [zoltar.distributions FlooredDistribution]
	   [zoltar.distributions SandpileDistribution]))

(deftest test-floored-distribution
  (testing
    (is (= (prob (FlooredDistribution. {} 0.123 0) "nothing") 0.123))
    (is (= (prob (FlooredDistribution. {:a 1} 0.01 0) :a) 1))
    (is (= (prob (-> (FlooredDistribution. {} 0.01 0)
		     (add-point :a 1)) :a) 1))
    (is (= (prob (-> (FlooredDistribution. {} 0.01 0)
		     (add-point :a 1)
		     (add-point :b 1)
		     (add-point :c 1)
		     (add-point :b 1)) :b) 1/2))
    (is (= (prob (-> (FlooredDistribution. {} 0.0002 0)
		     (add-point :a 1)
		     (add-point :b 1)
		     (add-point :c 1)) :d) 0.0002))
    (is (= (prob (-> (FlooredDistribution. {} 0.0001 0)
		     (add-point :a 1)
		     (add-point :b 2)
		     (add-point :c 3)) :b) 1/3))))

(deftest test-linear-sandpile-distribution
  (testing
    (let [dist (-> (SandpileDistribution. {} 0.03 2 0 identity)
		   (add-point 15 1)
		   (add-point 10 1))]
      (is (= (prob dist 15) (prob dist 10)))
      (is (= (prob dist 14) (prob dist 16)))
      (is (= (prob dist 13) (prob dist 17)))
      (is (= (prob dist 9) (prob dist 11)))
      (is (= (prob dist 8) (prob dist 12)))
      (is (= (prob dist 14) (prob dist 9)))
      (is (= (prob dist 13) (prob dist 8)))
      (is (= (prob dist :a) 0.03))
      (is (= (prob dist :a) (prob dist 18) (prob dist 7)))
      (is (= (+ (prob dist 13) (prob dist 15)) (* 2 (prob dist 14)))))))

(deftest test-exponential-sandpile-distribution
  (testing
    (let [dist (-> (SandpileDistribution. {} 0.007 3 0 pow2)
                   (add-point 52 1)
		   (add-point 10 1))]
      (is (= (prob dist 52) (prob dist 10)))
      (is (= (prob dist 51) (prob dist 53) (prob dist 9) (prob dist 11)))
      (is (= (prob dist 50) (prob dist 54) (prob dist 8) (prob dist 12)))
      (is (= (prob dist 49) (prob dist 55) (prob dist 7) (prob dist 13)))
      (is (= (prob dist 48) (prob dist 4) (prob dist 1000) 0.007))
      (is (= (* 2 (prob dist 49) (prob dist 50))))
      (is (= (* 2 (prob dist 50) (prob dist 51))))
      (is (= (* 2 (prob dist 51) (prob dist 52)))))))

(deftest test-bump-map
  (testing
    (is (= { 4 1  5 2  6 3  7 2  8 1 }
	   (bump-map {} 6 2 identity)))
    (is (= { 4 3  5 6  6 9  7 6  8 3 }
	   (bump-map {} 6 2 (partial * 3))))
    (is (= { 701 1 }
	   (bump-map {} 701 0 identity)))
    (is (= { 4 1  5 2  6 3  7 3  8 3  9 3  10 2  11 1 }
	   (-> {}
	       (bump-map 6 2 identity)
	       (bump-map 9 2 identity))))))

(deftest test-classify
  (testing
    (let [short-sample [{:sample "a" :category :short}
			{:sample "fasdfsa" :category :long}]
	  long-sample [{:sample "a"      :category :short}
		       {:sample "ba"     :category :short}
                       {:sample "xx"     :category :short}
		       {:sample "y"      :category :short}
		       {:sample "sdfsa"  :category :long}
		       {:sample "asdfda" :category :long}
		       {:sample "ssssa"  :category :long}
		       {:sample "asdfda" :category :long}]]
      (is (= :long
	     (-> (naive-bayes-model)
		 (train short-sample)
		 (classify "abcdefg"))))
      (is (= :short
	     (-> (naive-bayes-model)
		 (train long-sample)
		 (classify "ad"))))
      (is (= :long
	     (-> (boosted-bayes-model 10)
		 (train long-sample)
		 (classify "dafasdf")))))))

(deftest test-normalize
  (testing
    (is (= 1 (reduce + (normalize [1 1 1 1 1 1]))))
    (is (= 1 (reduce + (normalize [0.01 0.02 0.03 0.05]))))
    (let [original [23 1 20 14]
	  normalized (normalize original)]
      (is (= 1 (reduce + normalized)))
      (is (= (map / normalized original))))))

(deftest test-boostable-bayes
  (testing
    (is (= :C
	     (-> (boosted-bayes [{:category :A :sample [12]}
				 {:category :B :sample [2]}
				 {:category :C :sample [100]}]
				[1 1 1])
		 (classify [100]))))))