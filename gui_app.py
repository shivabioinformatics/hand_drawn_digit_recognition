# gui app to draw and classify hand-written digits
# using tkinter for GUI and PIL for drawing

# digit_drawer.py

import pygame
import sys

# Config
WINDOW_SIZE = 280  # 10x scale of 28x28
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DRAW_RADIUS = 10  # size of brush

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Draw a Digit")
    clock = pygame.time.Clock()

    screen.fill(BLACK)

    drawing = False

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
                    pygame.image.save(screen, "digit.png")
                    print("🖼️ Saved drawing as digit.png")
                elif event.key == pygame.K_c:
                    screen.fill(BLACK)
                    print("🧼 Canvas cleared")

        if drawing:
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.circle(screen, WHITE, mouse_pos, DRAW_RADIUS)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
