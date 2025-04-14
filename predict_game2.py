from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pygame, cv2

# Constante

CANVAS_SIZE = 64
DRAW_AREA_WIDTH = 300
BUTTON_AREA_WIDTH = 150
PREDICTION_AREA_WIDTH = 200
WINDOW_HEIGHT = DRAW_AREA_WIDTH + 100
WINDOW_WIDTH = DRAW_AREA_WIDTH + BUTTON_AREA_WIDTH + PREDICTION_AREA_WIDTH

TOP_PADDING, LEFT_PADDING = 20, 30
DRAW_AREA_TOP = TOP_PADDING + 50
DRAW_AREA_LEFT = LEFT_PADDING

BUTTON_X, BUTTON_Y = 2 * LEFT_PADDING + DRAW_AREA_WIDTH, DRAW_AREA_TOP
BUTTON_WIDTH = 260
BUTTON_HEIGHT = 50


# Culori

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLUE = (50, 50, 200)
LIGHT_BLUE = (100, 100, 255)
SHADOW = (0, 0, 0, 128)
YELLOW = (255, 255, 0)
RED = (255, 50, 50)


# functie cu ajutorul careia antrenam modelul : tryit_64.keras :


def train_model():
    # Încărcăm setul de date MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Redimensionăm imaginile la 64x64 și normalizăm valorile pixelilor
    x_train = tf.image.resize(x_train[..., tf.newaxis], [64, 64]) / 255.0
    x_test = tf.image.resize(x_test[..., tf.newaxis], [64, 64]) / 255.0

    # Construim modelul
    model = Sequential()

    # Blocul 1                           # Convoluție cu 32 de filtre
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(BatchNormalization())             # Normalizare pe loturi
    model.add(MaxPooling2D(pool_size=(2, 2)))   # Reducere dimensională (pooling)
    model.add(Dropout(0.3))                     # Eliminare aleatorie a neuronilor (regularizare)

    # Blocul 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Blocul 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Blocul 4
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Trecem la straturile dense (fully connected)
    model.add(GlobalAveragePooling2D())

    # Straturi Dense
    model.add(Dense(128, activation='relu'))  # Strat dens cu 128 de neuroni
    model.add(Dropout(0.5))  # Regularizare



    # Strat de ieșire
    model.add(Dense(10, activation='softmax'))  # 10 clase pentru cifre (0-9)

    # Compilarea modelului
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Afișează un rezumat al modelului
    model.summary()

    # Antrenarea modelului
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Evaluarea performanței pe setul de testare
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nAcuratețea pe setul de test:', test_acc)

    # Salvăm modelul antrenat
    model.save("digit_model.keras")


model = load_model('digit_model.keras')


pygame.init()

# Fereastra principală
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("NumPredict")

# Zona de desen
drawing_surface_rect = pygame.Rect(DRAW_AREA_LEFT, DRAW_AREA_TOP, DRAW_AREA_WIDTH, DRAW_AREA_WIDTH)
drawing_surface = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
drawing_surface.fill(WHITE)


# Afișează una sau mai multe imagini (pentru testare sau vizualizare)
def show_image(images):
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))

    for i, image in enumerate(images):
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'image {i + 1}')
        axes[i].axis('off')

    plt.show()

# Salvează conținutul zonei de desen ca imagine alb-negru

def save_canvas(surface, file_path="test_canvas.png"):
    arr = pygame.surfarray.array3d(surface)
    arr = np.transpose(arr, (1, 0, 2))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, binary_img = cv2.threshold(norm, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(file_path, binary_img)


# Prezice cifra desenată pe ecran
def predict_digit():
    # img = Image.open('test_canvas.png').convert('L')
    img = cv2.imread('test_canvas.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))

    # Convertește imaginea într-un array numpy și o normalizează în intervalul [0, 1]
    arr = img.astype(np.float32) / 255.0

    # Redimensionează array-ul pentru a adăuga dimensiuni pentru batch și canal
    arr = arr.reshape(1, 64, 64, 1)

    prediction = model.predict(arr)
    digit = np.argmax(prediction)
    confidence = prediction[0, digit]

    return digit, confidence


# Desenează butoanele "Predict" și "Clear"
def draw_buttons():
    font = pygame.font.Font(None, 50)

    # Butonul de predicție
    predict_rect = pygame.Rect(BUTTON_X, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT)
    pygame.draw.rect(screen, BLUE, predict_rect, border_radius=15)
    pygame.draw.rect(screen, LIGHT_BLUE, predict_rect, 4, border_radius=15)
    predict_text = font.render("Predict", True, WHITE)
    screen.blit(predict_text, predict_text.get_rect(center=predict_rect.center))

    # Butonul de ștergere
    clear_rect = pygame.Rect(BUTTON_X, BUTTON_Y + BUTTON_HEIGHT + 30, BUTTON_WIDTH, BUTTON_HEIGHT)
    pygame.draw.rect(screen, BLUE, clear_rect, border_radius=15)
    pygame.draw.rect(screen, LIGHT_BLUE, clear_rect, 4, border_radius=15)
    clear_text = font.render("Clear", True, WHITE)
    screen.blit(clear_text, clear_text.get_rect(center=clear_rect.center))

    return predict_rect, clear_rect


# Afișează cifra prezisă și nivelul de încredere
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

    # Afișăm textul în interiorul casetei de predicție
    y_offset = prediction_box.y + 10
    for line in text_lines:
        text_surface = font.render(line, True, WHITE)
        text_rect = text_surface.get_rect(center=(prediction_box.centerx, y_offset + 20))
        screen.blit(text_surface, text_rect)
        y_offset += 40


# Funcția care desenează tot ce apare pe ecran
def draw():
    # Titlul aplicației
    def draw_title():
        font = pygame.font.Font(None, 36)
        title_text = font.render("Handwritten Digit Recognizer", True, WHITE) # Recunoasterea Cifrelor cu Inteligenta Artificiala
        screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 20))

    # Fundalul cu gradient albastru
    def draw_gradient_background():
        for i in range(WINDOW_HEIGHT):
            color = (0, 0, int(255 * (i / WINDOW_HEIGHT)))  # Blue gradient
            pygame.draw.line(screen, color, (0, i), (WINDOW_WIDTH, i))

    # Cursorul personalizat (un cerc mic negru)
    def draw_cursor():
        x, y = pygame.mouse.get_pos()
        if y < WINDOW_HEIGHT:
            pygame.draw.circle(screen, BLACK, (x, y), 5)

    # Zona de desen cu colțuri rotunjite și imagine scalată
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


# Bucla principală a aplicației
running = True
predicted_digit, confidence = None, None

while running:
    screen.fill(GRAY)
    draw()

    # Desenăm butoanele
    predict_button, clear_button = draw_buttons()

    # Dacă există o predicție, o afișăm pe ecran
    if predicted_digit is not None and confidence is not None:
        display_prediction(predicted_digit, confidence)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if predict_button.collidepoint(event.pos):
                save_canvas(drawing_surface)                     # Salvăm desenul ca imagine
                predicted_digit, confidence = predict_digit()    # Prezicem cifra
            elif clear_button.collidepoint(event.pos):
                drawing_surface.fill(WHITE)                     # Curățăm desenul
                predicted_digit, confidence = None, None

        # Desenăm atunci când apăsăm click și mișcăm mouse-ul
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

"""

    # tryit64.keras || digit_model.keras        am schimbat numele pt poza


 if 0 <= x < DRAW_AREA_WIDTH and 0 <= y <  DRAW_AREA_WIDTH: # and 0 <= y < DRAW_AREA_WIDTH:
                draw_x = min(max(0, x * CANVAS_SIZE // DRAW_AREA_WIDTH), CANVAS_SIZE - 1)
                draw_y = min(max(0, y * CANVAS_SIZE // DRAW_AREA_WIDTH), CANVAS_SIZE - 1)
                pygame.draw.circle(drawing_surface, BLACK, (draw_x, draw_y), 2)

def draw_buttons2():
    # Draw Predict and Clear buttons in the middle area.
    font = pygame.font.Font(None, 36)

    # Predict Button
    predict_rect = pygame.Rect(DRAW_AREA_TOP + DRAW_AREA_WIDTH + 20, DRAW_AREA_TOP + 40 , BUTTON_AREA_WIDTH - 40, 60)
    pygame.draw.rect(screen, BLUE, predict_rect, border_radius=30)  # Rounded corners
    # button_surface = pygame.Surface(predict_rect.size, pygame.SRCALPHA)
    # button_surface.fill((50, 50, 200, 180))
    # screen.blit(button_surface, predict_rect.topleft)
    predict_text = font.render("Predict", True, WHITE)
    screen.blit(predict_text, predict_text.get_rect(center=predict_rect.center))

    # Clear Button
    clear_rect = pygame.Rect(DRAW_AREA_LEFT + DRAW_AREA_WIDTH + 20, DRAW_AREA_TOP + 140, BUTTON_AREA_WIDTH - 40, 60)
    pygame.draw.rect(screen, BLUE, clear_rect, border_radius=20)  # Rounded corners
    # button_surface = pygame.Surface(predict_rect.size, pygame.SRCALPHA)
    # button_surface.fill((50, 50, 200, 180))
    # screen.blit(button_surface, predict_rect.topleft)
    clear_text = font.render("Clear", True, WHITE)
    screen.blit(clear_text, clear_text.get_rect(center=clear_rect.center))

    return predict_rect, clear_rect

"""
