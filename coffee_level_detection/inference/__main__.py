import argparse
from coffee_level_detection.inference.tools import infer_coffee_level

def main():
	parser = argparse.ArgumentParser(description="Coffee Level Detection Inference CLI")
	parser.add_argument("--img", type=str, required=True, help="Path to input image")
	parser.add_argument("--model", type=str, default="coffeeCNN.pth", help="Path to trained model file")
	args = parser.parse_args()

	level = infer_coffee_level(args.img, args.model)
	print(f"Predicted coffee level: {level}")

if __name__ == "__main__":
	main()
