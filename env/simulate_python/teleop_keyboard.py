#!/usr/bin/env python3
import os
import pygame
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_ as WirelessController_type
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_ as WirelessController_msg

TOPIC_WIRELESS = "rt/wirelesscontroller"
DOMAIN_ID = 1
IFACE = "lo"
HZ = 100.0
AXIS_MAG = 0.9

KEYS = {
    "UP": pygame.K_UP,
    "DOWN": pygame.K_DOWN,
    "LEFT": pygame.K_LEFT,
    "RIGHT": pygame.K_RIGHT,
    "TURN_L": pygame.K_q,
    "TURN_R": pygame.K_e,
}

def get_axis(keys):
    lx = float(keys[KEYS["RIGHT"]] - keys[KEYS["LEFT"]]) * AXIS_MAG
    ly = float(keys[KEYS["UP"]] - keys[KEYS["DOWN"]]) * AXIS_MAG
    rx = float(keys[KEYS["TURN_R"]] - keys[KEYS["TURN_L"]]) * AXIS_MAG
    ry = 0.0
    return lx, ly, rx, ry

def main():
    ChannelFactoryInitialize(DOMAIN_ID, IFACE)
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    pygame.init()
    
    try:
        screen = pygame.display.set_mode((400, 200))
        pygame.display.set_caption("Keyboard → WirelessController")
        font = pygame.font.Font(None, 24)
    except pygame.error:
        screen = None
        font = None
    
    print("="*50)
    print("Keyboard Controller")
    print("="*50)
    print("  ↑/↓      - Forward/Backward")
    print("  ←/→      - Strafe Left/Right")
    print("  Q/E      - Turn Left/Right")
    print("  ESC      - Quit")
    print("="*50)
    
    pub = ChannelPublisher(TOPIC_WIRELESS, WirelessController_type)
    pub.Init()
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            break
        
        lx, ly, rx, ry = get_axis(keys)
        
        msg = WirelessController_msg()
        msg.lx = float(lx)
        msg.ly = float(ly)
        msg.rx = float(rx)
        msg.ry = float(ry)
        msg.keys = 0
        pub.Write(msg)
        
        # 显示状态
        if screen is not None and font is not None:
            screen.fill((30, 30, 30))
            
            texts = [
                "Walking Controller",
                f"Forward/Back: {ly:+.2f}",
                f"Left/Right:   {lx:+.2f}",
                f"Turn:         {rx:+.2f}",
            ]
            
            for i, text in enumerate(texts):
                color = (0, 255, 0) if i == 0 else (200, 200, 200)
                surf = font.render(text, True, color)
                screen.blit(surf, (10, 10 + i * 30))
            
            pygame.display.flip()
        
        clock.tick(HZ)
    
    pygame.quit()
    print("\nKeyboard controller stopped.")

if __name__ == "__main__":
    main()