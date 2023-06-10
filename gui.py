import machineLearning as magic
import pygame
from PIL import Image, ImageFilter
import numpy as np

# Configuration
classPath = "data/airbusboeing.txt"                 # Path to the file containing the classifications
networkPath = "data/network_64-airbusboeing-f.dat"  # Path to the network data file

# Uncustomisable variables
width = 64              # [px]
height = 32             # [px]
ratio = 2               # [-] Aspect ratio of the input images (width/height)
targetWidth = 64        # [px]
displayedAnswers = 13   # [-] Number of secondary classification answers outputted

families = []
with open(classPath) as f:
    lines = f.read().split("\n")
    for line in lines:
        if line not in families and len(line) > 0:
            families.append(line)

imageSurface = None
imageProcessed = None

def imageToSurface(_img):
    return pygame.image.fromstring(_img.tobytes(), _img.size, _img.mode).convert()

authors = 'Mike Kuijper and Koos Goudswaard'
creationDate = '2023-06-10'

titleCoords = [50, 10]

inputCoords = [50, 80]
processedCoords = [50, 400]
weightsCoords = []
outputCoords = [650, 130]

inputSize = [550, 275]
processedSize = [550, 275]
weightsSize = []
outputSize = [550, 425]

rectThickness = 5

pygame.init()
pygame.font.init()
basic_font = pygame.font.SysFont('Times New Roman', 30)
super_font = pygame.font.SysFont('Times New Roman', 40)
small_font = pygame.font.SysFont('Times New Roman', 20)
big_font = pygame.font.SysFont('Standard', 60)

screen = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("AE1205 AI Competition 2023 - Aircraft Classifier. Loading " + networkPath)
clock = pygame.time.Clock()
running = True

neuralNetworkInstance = magic.NeuralNetwork(networkPath)

outputHeadingText = "Output"
aircraftTypeText = ""
confidenceText = ""
otherAnswers = []
otherConfidences = []

# Edge detection kernel, horizontal Sobel operator. Applying this convolution to the image outputs the edge detected image.
edgeKernel = [1, 2, 1,
              0, 0, 0,
              -1, -2, -1]
kernel_filter = ImageFilter.Kernel((3, 3), edgeKernel, scale=1, offset=0)

pygame.display.set_caption("AE1205 AI Competition 2023 - Aircraft Classifier")
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.DROPFILE:
            image = Image.open(event.file)
            pilImageProcessed = image

            # Crop image to 2:1 (w:h) ratio
            if image.height * ratio > image.width:
                maxHeight = min(image.width / ratio, image.height)
                offset = int((image.height - maxHeight) / 2)

                pilImageProcessed = pilImageProcessed.crop((0, offset, image.width, image.height - offset))
            else:
                maxWidth = min(image.height * ratio, image.width)
                offset = int((image.width - maxWidth) / 2)

                pilImageProcessed = pilImageProcessed.crop((offset, 0, image.width - offset, image.height))

            imageSurface = imageToSurface(pilImageProcessed)

            pilImageProcessed = pilImageProcessed.filter(kernel_filter)
            pilImageProcessed = pilImageProcessed.resize((targetWidth, int(targetWidth / ratio)))

            imageProcessed = imageToSurface(pilImageProcessed)

            imageSurface = pygame.transform.smoothscale(imageSurface, (inputSize[0], inputSize[1]))

            imageProcessed = pygame.transform.scale(imageProcessed, (inputSize[0], inputSize[1]))
            array_imageProcessed = pygame.surfarray.array3d(imageProcessed)
            pixels_imageProcessed = [[(r * 299 / 1000 + g * 587 / 1000 + b * 114 / 1000) for (r, g, b) in col] for col in array_imageProcessed]
            # Convert pixels to grayscale. Same as PIL's method (https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert)

            array_imageProcessed = np.array([[[avg, avg, avg] for avg in col] for col in pixels_imageProcessed])
            imageProcessed = pygame.surfarray.make_surface(array_imageProcessed)

            vectorImage = pilImageProcessed.convert('L')
            pixels = [str(vectorImage.getpixel((x, y))) for y in range(0, int(targetWidth / ratio)) for x in range(0, targetWidth)]
            inputVector = magic.Vector(width*height).fromList(pixels)
            inputVector.setEach(lambda y, x, val: val/255)  # Normalise pixel values

            result = neuralNetworkInstance.feed(inputVector)
            resultList = result.toList()
            sortedIndex = sorted(range(len(resultList)), key=lambda k: -resultList[k])
            maxIndex = max(enumerate(resultList), key=lambda x: x[1])[0]
            confidenceText = str(round(resultList[maxIndex] * 100, 1)) + '%'

            aircraftTypeText = families[sortedIndex[0]]
            otherAnswers = [families[i] for i in sortedIndex[1:displayedAnswers]]
            otherConfidences = [str(round(resultList[i] * 100, 1)) for i in sortedIndex[1:displayedAnswers]]
            del result
            del resultList

    screen.fill("#302e30")
    text_title = basic_font.render('Aircraft classifier', True, (255, 255, 255))
    screen.blit(text_title, (titleCoords[0], titleCoords[1]))

    text_outputInstr = super_font.render(outputHeadingText, True, (255, 255, 255))
    screen.blit(text_outputInstr, (outputCoords[0] + outputSize[0] * 0.4, outputCoords[1] - 60))

    text_output = big_font.render(aircraftTypeText, True, (255, 255, 255))
    screen.blit(text_output, (outputCoords[0] + 10, outputCoords[1] + 15))

    text_confidence = big_font.render(confidenceText, True, (255, 255, 255))
    screen.blit(text_confidence, (outputCoords[0] + 400, outputCoords[1] + 15))

    for i in range(len(otherAnswers)):
        answer = otherAnswers[i]
        text_outputs = small_font.render(answer, True, (255, 255, 255))
        screen.blit(text_outputs, (outputCoords[0] + 10, outputCoords[1] + 65 + i * 30))

        _confidence = otherConfidences[i]
        text_confidences = small_font.render(_confidence + "%", True, (255, 255, 255))
        screen.blit(text_confidences, (outputCoords[0] + 400, outputCoords[1] + 65 + i * 30))

    text_inputInstruction = super_font.render('Drop your image!', True, (255, 255, 255))
    screen.blit(text_inputInstruction, (inputCoords[0] + 110, inputCoords[1] + inputSize[1] / 2 - 30))

    text_inputInfo = small_font.render('Input image:', True, (255, 255, 255))
    screen.blit(text_inputInfo, (inputCoords[0], inputCoords[1] - 30))

    text_processedInfo = small_font.render('Processed image:', True, (255, 255, 255))
    screen.blit(text_processedInfo, (processedCoords[0], processedCoords[1] - 30))

    text_authors = basic_font.render('by ' + authors, True, (255, 255, 255))
    screen.blit(text_authors, (outputCoords[0], outputCoords[1] + outputSize[1] + 50))

    text_date = small_font.render(creationDate, True, (255, 255, 255))
    screen.blit(text_date, (outputCoords[0], outputCoords[1] + outputSize[1] + 100))

    # rectangles to draw around the images and text as boxes
    r1 = pygame.Rect(inputCoords[0] - rectThickness,
                     inputCoords[1] - rectThickness,
                     inputSize[0] + 2 * rectThickness,
                     inputSize[1] + 2 * rectThickness)
    pygame.draw.rect(screen, "white", r1, width=rectThickness)

    r2 = pygame.Rect(processedCoords[0] - rectThickness,
                     processedCoords[1] - rectThickness,
                     processedSize[0] + 2 * rectThickness,
                     processedSize[1] + 2 * rectThickness)
    pygame.draw.rect(screen, "white", r2, width=rectThickness)

    r3 = pygame.Rect(outputCoords[0] - rectThickness,
                     outputCoords[1] - rectThickness,
                     outputSize[0] + 2 * rectThickness,
                     outputSize[1] + 2 * rectThickness)
    pygame.draw.rect(screen, "white", r3, width=rectThickness)

    if imageSurface is not None:
        screen.blit(imageSurface, (inputCoords[0], inputCoords[1]))
        screen.blit(imageProcessed, (processedCoords[0], processedCoords[1]))

    pygame.display.flip()
    clock.tick(10)  # limits FPS to 10

del neuralNetworkInstance   # Burn the witch!
