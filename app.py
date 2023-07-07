from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel
from pickle import TRUE
from flask import Flask, render_template, request
from pyspark.ml.regression import DecisionTreeRegressionModel
from pyspark.sql.types import StructField, StructType, IntegerType
from pyspark import SparkContext
from collections.abc import MutableMapping
import pandas

#from collections import MutableMapping

sc = SparkContext()


app = Flask(__name__)


@app.route('/')
def man():
    return render_template('index.html')

@app.route('/predicted')
def predicted():
    return render_template('home.html')


@app.route('/north')
def north():
    return render_template("north.html")


@app.route('/south')
def south():
    return render_template("south.html")


@app.route('/east')
def east():
    return render_template("east.html")


@app.route('/west')
def west():
    return render_template("west.html")


@app.route('/predict', methods=['POST'])
def home():
    da = request.form['a']
    data2 = int(request.form['Hour'])
    data3 = int(request.form['OS'])
    data1 = int(request.form['MONTH'])
    data4 = int(request.form['RegionIndex'])
    data5 = int(request.form['InstanceIndex'])
    acdata = sc.parallelize(
        [{'Month': data1, 'Hour': data2, 'OS': data3, 'Region': data4, 'Instance': data5}])
    reg1 = ['north','south','east','west']
    os1 = ['Windows','Linux/UNIX']
    instan = ["p2.xlarge","r3.xlarge","c4.xlarge","r4.8xlarge","g2.2xlarge","m3.2xlarge","c4.4xlarge","c3.2xlarge","r3.4xlarge","r3.large","cc2.8xlarge","m3.xlarge","d2.8xlarge","d2.2xlarge","c3.8xlarge","m3.large","i3.8xlarge","m4.16xlarge","r3.2xlarge","i2.xlarge","c1.medium","m4.4xlarge","c4.2xlarge","c1.xlarge","c4.8xlarge","m4.2xlarge","x1.16xlarge","m4.2xlarge","i3.2xlarge","r3.8xlarge","i3.xlarge","p2.8xlarge","m3.medium","c3.4xlarge","m4.10xlarge","r4.4xlarge","m1.large","m1.medium","i2.4xlarge","r4.16xlarge","m4.large","r4.16xlarge","c3.large","r4.xlarge","i2.2xlarge","m2.xlarge","x1.32xlarge","m4.xlarge","m1.xlarge","r4.large","cr1.8xlarge","c3.xlarge","hi1.4xlarge","m2.4xlarge","g2.8xlarge","d2.xlarge","i3.16xlarge","d2.4xlarge","c4.large","m2.2xlarge","i2.8xlarge","i3.large","t1.micro","i3.4xlarge","p2.16xlarge","m1.small","cg1.4xlarge"]
    schema = StructType([
        StructField('Month', IntegerType(), True),
        StructField('Hour', IntegerType(), True),
        StructField('OS', IntegerType(), True),
        StructField("Region", IntegerType(), True),
        StructField("Instance", IntegerType(), True)
    ])
    from pyspark import SQLContext
    sqlContext = SQLContext(sc)
    df = sqlContext.createDataFrame(acdata, schema)
    cols0 = ["Month", "Hour", "OS", "Region", "Instance"]
    vec_assm = VectorAssembler(inputCols=cols0, outputCol='features')
    test = vec_assm.transform(df)
    if da == "north":
        model_1 = LinearRegressionModel.load(
            "LinearReg/")

        pred = model_1.transform(test)
        p = pred.select("prediction")
        c = p.toPandas()
        a = c.prediction.values[0]
        return render_template('after.html', msg=a , reg=reg1[data4], os = os1[data3], inst=instan[data5])


if __name__ == "__main__":
    app.run(debug=TRUE)
