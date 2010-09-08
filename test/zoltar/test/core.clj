(ns zoltar.test.core
  (:use [zoltar.core] :reload)
  (:use [clojure.test])
  (:import [zoltar.core FlooredDistribution]))

(deftest test-floor
  (testing
    (is (= (prob (FlooredDistribution. {} 0.123) "nothing") 0.123))
    (is (= (prob (FlooredDistribution. {:a 1} 0.01) :a) 1))))

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
      (is (= "long"
	     (-> (naive-bayes-model)
		 (train "a" "short")
		 (train "fasdfsa" "long")
		 (classify "abcdefg"))))))