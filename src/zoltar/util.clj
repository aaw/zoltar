(ns ^{:author "Aaron Windsor",
      :doc "Utilities for reading test data"}
  zoltar.util
  (:use zoltar.classifiers))

(defn parse-letters-line [^String line]
  (let [line-seq (seq (.split line ","))]
    {:category (first line-seq)
     :sample (map #(Integer/parseInt %) (rest line-seq))}))

(defn parse-iris-line [^String line]
  (let [line-seq (seq (.split line ","))
	clean-num (comp #(int %) #(* 10 %) #(Double/parseDouble %))]
    {:category (last line-seq)
     :sample (map clean-num (drop-last line-seq))}))

(defn read-data [^String filename parse-line]
  (with-open [reader (java.io.BufferedReader. (java.io.FileReader. filename))]
    (let [lseq (line-seq reader)]
      (vec (map parse-line lseq)))))

(def letters-datafile "datasets/letter-recognition.data.txt")
(def iris-datafile "datasets/iris.data.txt")

(def letters-dataset (read-data letters-datafile parse-letters-line))
(def iris-dataset (shuffle (read-data iris-datafile parse-iris-line)))

(def datasets { :letters letters-dataset :iris iris-dataset })

(defn k-fold-cross-validation [model data k]
  (let [partitions (partition-all k data)
	num-partitions (count partitions)
	exclude-nth (fn [s n] (flatten (concat (take n s) (drop (inc n) s))))
	score (fn [model samples] (/ (reduce + (map indicator (repeat model) samples)) (count samples)))
	avg (fn [xs] (/ (apply + xs) (count xs)))
	kth-score (fn [k] (-> model (train (exclude-nth partitions k)) (score (nth partitions k))))]
    (avg (map kth-score (range num-partitions)))))