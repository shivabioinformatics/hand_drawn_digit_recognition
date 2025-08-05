# gui app to draw and classify hand-written digits
# using tkinter for GUI and PIL for drawing

import pygame
import sys
import numpy as np
from predict import predict_digit_from_array

WINDOW_SIZE = 500
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DRAW_RADIUS = 8

def predict_from_surface(surface):
    surface_str = pygame.image.tostring(surface, 'RGB')
    w, h = surface.get_size()
    img = np.frombuffer(surface_str, dtype=np.uint8).reshape((h, w, 3))
    return predict_digit_from_array(img)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Draw a Digit")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 30)

    screen.fill(BLACK)
    drawing = False

    last_prediction = None
    last_confidence = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True

            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    prediction, confidence = predict_from_surface(screen)
                    last_prediction = int(prediction)
                    last_confidence = float(confidence)

                elif event.key == pygame.K_c:
                    screen.fill(BLACK)
                    last_prediction = None
                    last_confidence = None

        if drawing:
            pos = pygame.mouse.get_pos()
            pygame.draw.circle(screen, WHITE, pos, DRAW_RADIUS)

        # Draw the prediction text if available
        if last_prediction is not None and last_confidence is not None:
            text_surface = font.render(
                f"Predicted: {last_prediction} (Confidence: {last_confidence:.2f})", True, WHITE)
            screen.blit(text_surface, (5, WINDOW_SIZE - 40))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
