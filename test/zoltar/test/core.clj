(ns zoltar.test.core
  (:use [zoltar.core] :reload)
  (:use [zoltar.distributions] :reload)
  (:use [zoltar.feature_testers] :reload)
  (:use [clojure.test])
  (:import [zoltar.distributions FlooredDistribution]
	   [zoltar.distributions LinearSandpileDistribution]
	   [zoltar.distributions ExponentialSandpileDistribution]))

(deftest test-floored-distribution
  (testing
    (is (= (prob (FlooredDistribution. {} 0.123) "nothing") 0.123))
    (is (= (prob (FlooredDistribution. {:a 1} 0.01) :a) 1))
    (is (= (prob (add-point (FlooredDistribution. {} 0.01) :a) :a) 1))
    (is (= (prob (-> (FlooredDistribution. {} 0.01)
		     (add-point :a)
		     (add-point :b)
		     (add-point :c)
		     (add-point :b)) :b) 1/2))
    (is (= (prob (-> (FlooredDistribution. {} 0.0002)
		     (add-point :a)
		     (add-point :b)
		     (add-point :c)) :d) 0.0002))))

(deftest test-linear-sandpile-distribution
  (testing
    (let [dist (-> (LinearSandpileDistribution. {} 0.03 2)
		   (add-point 15)
		   (add-point 10))]
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
    (let [dist (-> (ExponentialSandpileDistribution. {} 0.007 3)
                   (add-point 52)
		   (add-point 10))]
      (is (= (prob dist 52) (prob dist 10)))
      (is (= (prob dist 51) (prob dist 53) (prob dist 9) (prob dist 11)))
      (is (= (prob dist 50) (prob dist 54) (prob dist 8) (prob dist 12)))
      (is (= (prob dist 49) (prob dist 55) (prob dist 7) (prob dist 13)))
      (is (= (prob dist 48) (prob dist 4) (prob dist 1000) 0.007))
      (is (= (* 2 (prob dist 49) (prob dist 50))))
      (is (= (* 2 (prob dist 50) (prob dist 51))))
      (is (= (* 2 (prob dist 51) (prob dist 52)))))))
	
(deftest test-training
  (testing
    (is (= 1/3
	   (-> { :dist (FlooredDistribution. {} 0.0) :testfunc count }
	       (train-tester "aaa")
	       (train-tester "abc")
	       (train-tester "adss")
	       (test-tester "1234"))))
    (is (= 0.001
	   (-> { :dist (FlooredDistribution. {} 0.001) :testfunc count }
	       (train-tester "a")
	       (train-tester "aa")
	       (train-tester "aaa")
	       (train-tester "aaaa")
	       (test-tester "aaaaa"))))))

(deftest test-classify
  (testing
      (is (= :long
	     (-> (naive-bayes-model)
		 (train "a" :short)
		 (train "fasdfsa" :long)
		 (classify "abcdefg"))))))