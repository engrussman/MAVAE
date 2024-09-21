
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_accuracy(a,b):
	sum(1 for x,y in zip(a,b) if x == y) / len(a)


def print_quant_measures(Predicted_age,Inferenced_age,Predicted_Gender,Inferenced_Gender):
	print(get_accuracy(Predicted_age,Inferenced_age))
	print(get_accuracy(Predicted_Gender,Inferenced_Gender))
	print(f1_score(Predicted_age,Inferenced_age))
	print(f1_score(Predicted_Gender,Inferenced_Gender))
	print(precision_score(Predicted_age,Inferenced_age))
	print(precision_score(Predicted_Gender,Inferenced_Gender))
	print(recall_score(Predicted_age,Inferenced_age))
	print(recall_score(Predicted_Gender,Inferenced_Gender))