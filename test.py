import cv2
# from genetic_algorithm.GA import GA
# from genetic_algorithm.IGA import GA
from IGA_v3 import GA
# from genetic_algorithm.OGA import GA
from otsu import otsuth

if __name__ == '__main__':
    file_path = 'Pyramid.jpg'
    image = cv2.imread(file_path)

    cross_pro = 0.5
    mutation_pro = 0.001

    model = GA(image, otsuth, 100, 1000, crossover_pro=cross_pro, variation_pro=mutation_pro)
    model.run()

