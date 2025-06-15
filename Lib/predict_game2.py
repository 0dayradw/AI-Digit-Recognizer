from Lib.constants import *
import pygame
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

def run_prediction_app():

    model = load_model('model/digit_model.keras')

    pygame.init()

    # Main window
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("NumPredict")

    # images area 
    drawing_surface_rect = pygame.Rect(DRAW_AREA_LEFT, DRAW_AREA_TOP, DRAW_AREA_WIDTH, DRAW_AREA_WIDTH)
    drawing_surface = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
    drawing_surface.fill(WHITE)


    # Helper function to see multiple images
    def show_image(images):
        fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))

        for i, image in enumerate(images):
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'image {i + 1}')
            axes[i].axis('off')

        plt.show()

    # After drawing we save as a photo and give it to the model for prediction
    def save_canvas(surface, file_path="images/test_canvas.png"):
        arr = pygame.surfarray.array3d(surface)
        arr = np.transpose(arr, (1, 0, 2))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        _, binary_img = cv2.threshold(norm, 127, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(file_path, binary_img)


    # Function to predict on an image given
    def predict_digit():
        img = cv2.imread('images/test_canvas.png', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))

        arr = img.astype(np.float32) / 255.0
        arr = arr.reshape(1, 64, 64, 1)

        prediction = model.predict(arr)
        digit = np.argmax(prediction)
        confidence = prediction[0, digit]

        return digit, confidence


    # Funtion for drawing the buttons
    def draw_buttons():
        font = pygame.font.Font(None, 50)

        # Prediction button
        predict_rect = pygame.Rect(BUTTON_X, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT)
        pygame.draw.rect(screen, BLUE, predict_rect, border_radius=15)
        pygame.draw.rect(screen, LIGHT_BLUE, predict_rect, 4, border_radius=15)
        predict_text = font.render("Predict", True, WHITE)
        screen.blit(predict_text, predict_text.get_rect(center=predict_rect.center))

        # Clear button 
        clear_rect = pygame.Rect(BUTTON_X, BUTTON_Y + BUTTON_HEIGHT + 30, BUTTON_WIDTH, BUTTON_HEIGHT)
        pygame.draw.rect(screen, BLUE, clear_rect, border_radius=15)
        pygame.draw.rect(screen, LIGHT_BLUE, clear_rect, 4, border_radius=15)
        clear_text = font.render("Clear", True, WHITE)
        screen.blit(clear_text, clear_text.get_rect(center=clear_rect.center))

        return predict_rect, clear_rect


    # We output the predicted digit
    def display_prediction(digit, confidence):
        font = pygame.font.Font(None, 50)
        text_lines = [
            "Prediction:",
            f"Digit: {digit}",
            f"Conf: {confidence:.2%}"
        ]
        BOX_X, BOX_Y = BUTTON_X, BUTTON_Y + BUTTON_HEIGHT + 30 + BUTTON_HEIGHT + 30
        prediction_box = pygame.Rect(BOX_X, BOX_Y, BUTTON_WIDTH, 2 * BUTTON_HEIGHT + 40)

        pygame.draw.rect(screen, BLUE, prediction_box, border_radius=15)
        pygame.draw.rect(screen, LIGHT_BLUE, prediction_box, 4, border_radius=15)

        y_offset = prediction_box.y + 10
        for line in text_lines:
            text_surface = font.render(line, True, WHITE)
            text_rect = text_surface.get_rect(center=(prediction_box.centerx, y_offset + 20))
            screen.blit(text_surface, text_rect)
            y_offset += 40


    # function to draw everything on screen
    def draw():
        # app title
        def draw_title():
            font = pygame.font.Font(None, 36)
            title_text = font.render("Handwritten Digit Recognizer", True, WHITE)
            screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 20))

        # background with a dark blue gradient 
        def draw_gradient_background():
            for i in range(WINDOW_HEIGHT):
                color = (0, 0, int(255 * (i / WINDOW_HEIGHT)))  # Blue gradient
                pygame.draw.line(screen, color, (0, i), (WINDOW_WIDTH, i))

        # personalized cursor
        def draw_cursor():
            x, y = pygame.mouse.get_pos()
            if y < WINDOW_HEIGHT:
                pygame.draw.circle(screen, BLACK, (x, y), 5)

        # Drawing area 
        def draw_rounded_drawing_surface():
            rounded_surface = pygame.Surface((DRAW_AREA_WIDTH, DRAW_AREA_WIDTH), pygame.SRCALPHA)
            pygame.draw.rect(rounded_surface, WHITE, (0, 0, DRAW_AREA_WIDTH, DRAW_AREA_WIDTH), border_radius=20)

            scaled_surface = pygame.transform.scale(drawing_surface, (DRAW_AREA_WIDTH, DRAW_AREA_WIDTH))
            rounded_surface.blit(scaled_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

            screen.blit(rounded_surface, (DRAW_AREA_LEFT, DRAW_AREA_TOP))

        draw_gradient_background()
        draw_rounded_drawing_surface()
        draw_title()
        draw_cursor()

    running = True
    predicted_digit, confidence = None, None

    while running:
        screen.fill(GRAY)
        draw()

        # We draw the buttons on screen
        predict_button, clear_button = draw_buttons()

        # If there is a prediction we show it 
        if predicted_digit is not None and confidence is not None:
            display_prediction(predicted_digit, confidence)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if predict_button.collidepoint(event.pos):
                    save_canvas(drawing_surface)                     # we save the canvas 
                    predicted_digit, confidence = predict_digit()    # and we predict on it
                elif clear_button.collidepoint(event.pos):
                    drawing_surface.fill(WHITE)                    
                    predicted_digit, confidence = None, None

            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                x, y = pygame.mouse.get_pos()
                if drawing_surface_rect.collidepoint(x, y):
                    rel_x = x - DRAW_AREA_LEFT
                    rel_y = y - DRAW_AREA_TOP
                    draw_x = int(rel_x * CANVAS_SIZE / DRAW_AREA_WIDTH)
                    draw_y = int(rel_y * CANVAS_SIZE / DRAW_AREA_WIDTH)
                    pygame.draw.circle(drawing_surface, BLACK, (draw_x, draw_y), 2)

        pygame.display.flip()

    pygame.quit()
