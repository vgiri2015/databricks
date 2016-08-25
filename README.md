// Databricks notebook source exported at Thu, 25 Aug 2016 05:40:03 UTC
// MAGIC %md #Using Stanford CoreNLP with Apache Spark (Works Only in Spark 1.6.1 Hadoop 1 or 2)
// MAGIC 
// MAGIC Author: Xiangrui Meng
// MAGIC 
// MAGIC This is an example notebook for the [Stanford CoreNLP wrapper for Apache Spark](https://spark-packages.org/package/databricks/spark-corenlp) release v0.1.

// COMMAND ----------

// MAGIC %md ##Install Stanford CoreNLP wrapper for Apache Spark (spark-corenlp)

// COMMAND ----------

// MAGIC %md Databricks users should first add databricks:spark-corenlp:0.1 as a library and then attach it to a Spark 1.6 cluster to use.
// MAGIC The following code should run without any errors if the library is correctly added and attached.

// COMMAND ----------

import org.apache.spark.sql.functions._
import com.databricks.spark.corenlp.functions._

// COMMAND ----------

// MAGIC %md ##Install language models
// MAGIC 
// MAGIC In order to use advanced CoreNLP features, we need to install one of the language models and attach it to the cluster.
// MAGIC The models are packaged as jars.
// MAGIC Because they are quite large, exceeding the jar size limit on Databricks, we download the jar directly and add it to Spark context.
// MAGIC To get a list of supported models, please visit [CoreNLP website](http://stanfordnlp.github.io/CoreNLP/#human-languages-supported).

// COMMAND ----------

val version = "3.6.0"
val model = s"stanford-corenlp-$version-models" // append "-english" to use the full English model
val jars = sc.asInstanceOf[{def addedJars: scala.collection.mutable.Map[String, Long]}].addedJars.keys // use sc.listJars in Spark 2.0
if (!jars.exists(jar => jar.contains(model))) {
  import scala.sys.process._
  s"wget http://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/$version/$model.jar -O /tmp/$model.jar".!!
  sc.addJar(s"/tmp/$model.jar")
}

// COMMAND ----------

import edu.stanford.nlp.ie.crf._
import edu.stanford.nlp.ie.AbstractSequenceClassifier
import edu.stanford.nlp.ling.CoreLabel
import edu.stanford.nlp.ling.CoreAnnotations.AnswerAnnotation
import edu.stanford.nlp.util.StringUtils

// COMMAND ----------

// MAGIC %md ##Use CoreNLP functions as DataFrame functions
// MAGIC 
// MAGIC With `spark-corenlp`, users can use CoreNLP as DataFrame functions.

// COMMAND ----------

val input = Seq(
  (1, "<xml>Stanford University is located in California. It is a great university.</xml>")
).toDF("id", "text")

// COMMAND ----------

val output = input
  .select(cleanxml('text).as('doc))
  .select(explode(ssplit('doc)).as('sen))
  .select('sen, tokenize('sen).as('words), ner('sen).as('nerTags), sentiment('sen).as('sentiment))

// COMMAND ----------

display(output)

// COMMAND ----------

