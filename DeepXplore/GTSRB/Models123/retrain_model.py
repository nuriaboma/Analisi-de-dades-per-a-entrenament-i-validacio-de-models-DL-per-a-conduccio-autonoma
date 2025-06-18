
import argparse


parser = argparse.ArgumentParser(
    description='Create datasets to retrain models')
parser.add_argument('model', help="model used", choices=['Model1', 'Model2', 'Model3'])
parser.add_argument('retrain_data', help="Path to the data added for the retraining", type=str)
parser.add_argument('test_data', help="Path to the test data for the retraining", type=str)

args = parser.parse_args()


if args.model == 'Model1':
  from Model1 import Model1 as Model
elif args.model == 'Model2':
  from Model2 import Model2 as Model
elif args.model == 'Model3':
  from Model3 import Model3 as Model

model = Model(retrain=True, newData = args.retrain_data, newTestData=args.test_data)