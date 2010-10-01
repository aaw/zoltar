(ns ^{:author "Aaron Windsor",
      :doc "Utilities for reading test data"}
  zoltar.util)

(defn parse-file-line [^String line]
  (let [line-seq (seq (.split line ","))]
    {:category (first line-seq)
     :sample (map #(Integer/parseInt %) (rest line-seq))}))

(defn read-data [^String filename]
  (with-open [reader (java.io.BufferedReader. (java.io.FileReader. filename))]
    (let [lseq (line-seq reader)]
      (vec (map parse-file-line lseq)))))

(def letters-dataset (read-data "/home/aaron/development/machine-learning-datasets/letter-recognition.data.txt"))

