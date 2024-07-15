from confluent_kafka import Producer

def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

producer_conf = {
    "bootstrap.servers": "localhost:9092",
    "debug": "broker,topic,msg",
}
producer = Producer(producer_conf)

producer.produce('test_topic', key='key', value='value', callback=delivery_report)
producer.flush()
