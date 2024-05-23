import pygame
import pandas as pd
import random

file_path = 'output/union_for_gui/union_new.csv'
data = pd.read_csv(file_path, usecols=lambda col: col != 'permeability')

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
BACKGROUND_COLOR = (255, 255, 255)

AGENT_COLORS = {
    'lps_lumen': (255, 0, 0),
    'tnfAlfa': (0, 255, 0),
    'alfasin': (0, 0, 255),
    'scfa': (255, 255, 0),
    'lps_microbiota': (0, 255, 255)
}

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Gut GUI")

def draw_agents(agent_counts, tick):
    screen.fill(BACKGROUND_COLOR)

    for agent_type, count in agent_counts.items():
        agent_color = AGENT_COLORS.get(agent_type, (0, 0, 0))
        for _ in range(count):
            x = random.randint(0, SCREEN_WIDTH)
            y = random.randint(0, SCREEN_HEIGHT)
            pygame.draw.circle(screen, agent_color, (x, y), 4)

    draw_legend()

    font = pygame.font.Font(None, 36)
    text = font.render(f"Tick: {tick}", True, (0, 0, 0))
    screen.blit(text, (10, 10))

    pygame.display.flip()

def draw_legend():
    legend_x = 10
    legend_y = SCREEN_HEIGHT - 80
    font = pygame.font.Font(None, 24)

    for i, (agent_type, color) in enumerate(AGENT_COLORS.items()):
        pygame.draw.rect(screen, color, (legend_x, legend_y, 20, 20))

        legend_text = f"{agent_type}"
        text_surface = font.render(legend_text, True, (0, 0, 0))
        screen.blit(text_surface, (legend_x + 30, legend_y))

        legend_x += 200

        if (i + 1) % 3 == 0:
            legend_x = 10 
            legend_y += 40


running = True
index = 0
tick = 0
manual_tick_advance = 0

clock = pygame.time.Clock()
delay = 100

auto_mode = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:  # Press 'a' for automatic mode
                auto_mode = True
                manual_tick_advance = 0
            elif event.key == pygame.K_m:  # Press 'm' for manual mode
                auto_mode = False

            if not auto_mode:  # If we press 'm'
                if event.key == pygame.K_RIGHT:  # Right arrow
                    manual_tick_advance = 1
                elif event.key == pygame.K_LEFT:  # Left arrow
                    manual_tick_advance = -1

    if auto_mode:
        index = (index + 1) % len(data)
        tick += 1

        agent_counts = {
            'lps_lumen': data.iloc[index]['lps_lumen'],
            'tnfAlfa': data.iloc[index]['tnfAlfa'],
            'alfasin': data.iloc[index]['alfasin'],
            'scfa': data.iloc[index]['scfa'],
            'lps_microbiota': data.iloc[index]['lps_microbiota']
        }

        draw_agents(agent_counts, tick)

    else:
        if manual_tick_advance != 0:
            new_index = (index + manual_tick_advance) % len(data)
            if new_index != index:
                index = new_index
                tick += manual_tick_advance

                agent_counts = {
                    'lps_lumen': data.iloc[index]['lps_lumen'],
                    'tnfAlfa': data.iloc[index]['tnfAlfa'],
                    'alfasin': data.iloc[index]['alfasin'],
                    'scfa': data.iloc[index]['scfa'],
                    'lps_microbiota': data.iloc[index]['lps_microbiota']
                }
                
                draw_agents(agent_counts, tick)

            manual_tick_advance = 0

    pygame.display.update()
    clock.tick(500)
    pygame.time.delay(delay)

pygame.quit()