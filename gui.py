import os
import cv2
import pygame
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctkinter
ctinker = ctkinter
from PIL import Image, ImageTk
import time
from datetime import datetime
import threading

# Import project files
from config import Config
from utils import generate_default_alarm, save_screenshot, log_prediction, export_logs
from model_loader import ModelLoader
from face_detector import FaceDetector
from predictor import DrowsinessPredictor
from webcam import WebcamStream

# Set appearance mode and color theme
ctinker.set_appearance_mode("Dark")
ctinker.set_default_color_theme("blue")

class AlarmPlayer:
    """Helper class to play and loop the alarm sound using pygame."""
    def __init__(self, alarm_path, volume=0.8):
        self.alarm_path = alarm_path
        self.volume = volume
        self.initialized = False
        self.is_playing = False
        
        self.initialize()

    def initialize(self):
        try:
            pygame.mixer.init()
            if os.path.exists(self.alarm_path):
                self.sound = pygame.mixer.Sound(self.alarm_path)
                self.sound.set_volume(self.volume)
                self.initialized = True
            else:
                self.sound = None
                self.initialized = False
        except Exception as e:
            print(f"Error initializing Pygame Mixer: {e}")
            self.sound = None
            self.initialized = False

    def play(self):
        if not self.initialized:
            self.initialize()
        if self.initialized and self.sound and not self.is_playing:
            try:
                self.sound.play(loops=-1) # Loop forever
                self.is_playing = True
            except Exception as e:
                print(f"Error playing alarm sound: {e}")

    def stop(self):
        if self.initialized and self.sound and self.is_playing:
            try:
                self.sound.stop()
                self.is_playing = False
            except Exception as e:
                print(f"Error stopping alarm sound: {e}")

    def set_volume(self, volume):
        self.volume = volume
        if self.initialized and self.sound:
            try:
                self.sound.set_volume(self.volume)
            except Exception as e:
                print(f"Error setting volume: {e}")


class DDDApp(ctinker.CTk):
    """
    Main Application GUI Class. Inherits from CustomTkinter CTk.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Initialize Configuration
        self.config = Config()
        self.config.create_dirs()
        
        # Ensure default alarm sound exists
        generate_default_alarm(self.config.alarm_path)
        
        # 2. Main Window Properties
        self.title("DDD - Driver Drowsiness Detection")
        self.geometry("1200x750")
        self.resizable(True, True)
        self.minsize(1100, 700)
        
        # 3. Setup Components
        self.model_loader = ModelLoader()
        self.face_detector = FaceDetector(min_detection_confidence=0.5)
        self.predictor = DrowsinessPredictor(self.model_loader, self.config)
        self.webcam = None
        self.alarm_player = AlarmPlayer(self.config.alarm_path, self.config.alarm_volume)
        
        # 4. App States & Stats
        self.is_fullscreen = False
        self.camera_running = False
        
        # Statistics
        self.stat_total_frames = 0
        self.stat_total_drowsy_frames = 0
        self.stat_total_alarms = 0
        self.stat_today_detections = 0
        self.session_start_time = None
        self.alarm_start_time = None
        
        # For flashing warning screen
        self.flash_state = False
        
        # Build standard UI layout
        self.setup_ui_layout()
        
        # Show Splash Screen loading sequence at startup
        self.show_splash_screen()
        
        # Bind closing
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    def setup_ui_layout(self):
        """Creates the main visual grid and panels."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Top Header Frame with Indigo/Dark gradient theme
        self.header_frame = ctkinter.CTkFrame(self, height=80, corner_radius=0, fg_color="#1F1F2E")
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=1)
        
        header_title = ctkinter.CTkLabel(
            self.header_frame, 
            text="Driver Drowsiness Detection", 
            font=ctinker.CTkFont(family="Helvetica", size=24, weight="bold"),
            text_color="#FFFFFF"
        )
        header_title.pack(pady=(12, 2), anchor="center")
        
        header_sub = ctkinter.CTkLabel(
            self.header_frame, 
            text="AI Powered Driver Safety Monitoring System", 
            font=ctinker.CTkFont(family="Helvetica", size=13, slant="italic"),
            text_color="#A5A5C7"
        )
        header_sub.pack(anchor="center")
        
        # Left Navigation Sidebar Frame
        self.sidebar_frame = ctkinter.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=1, column=0, sticky="nsw")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)
        
        sidebar_lbl = ctkinter.CTkLabel(
            self.sidebar_frame, 
            text="NAVIGATION", 
            font=ctinker.CTkFont(family="Helvetica", size=12, weight="bold"),
            text_color="#777777"
        )
        sidebar_lbl.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Sidebar Buttons
        self.btn_dashboard = ctkinter.CTkButton(
            self.sidebar_frame, text="Dashboard", command=self.show_dashboard_page,
            fg_color="#1A237E", hover_color="#283593", height=40
        )
        self.btn_dashboard.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        
        self.btn_load_model = ctkinter.CTkButton(
            self.sidebar_frame, text="Load Model", command=self.load_model_dialog, height=40
        )
        self.btn_load_model.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        
        self.btn_start_cam = ctkinter.CTkButton(
            self.sidebar_frame, text="Start Camera", command=self.start_camera,
            fg_color="#2E7D32", hover_color="#1B5E20", height=40, state="disabled"
        )
        self.btn_start_cam.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        
        self.btn_stop_cam = ctkinter.CTkButton(
            self.sidebar_frame, text="Stop Camera", command=self.stop_camera,
            fg_color="#C62828", hover_color="#B71C1C", height=40, state="disabled"
        )
        self.btn_stop_cam.grid(row=4, column=0, padx=20, pady=5, sticky="ew")
        
        self.btn_settings = ctkinter.CTkButton(
            self.sidebar_frame, text="Settings", command=self.show_settings_page, height=40
        )
        self.btn_settings.grid(row=5, column=0, padx=20, pady=5, sticky="ew")
        
        self.btn_stats = ctkinter.CTkButton(
            self.sidebar_frame, text="Statistics", command=self.show_stats_page, height=40
        )
        self.btn_stats.grid(row=6, column=0, padx=20, pady=5, sticky="ew")
        
        self.btn_reconnect = ctkinter.CTkButton(
            self.sidebar_frame, text="Reconnect Cam", command=self.reconnect_camera, height=40
        )
        self.btn_reconnect.grid(row=7, column=0, padx=20, pady=5, sticky="ew")
        
        self.btn_exit = ctkinter.CTkButton(
            self.sidebar_frame, text="Exit", command=self.on_exit, 
            fg_color="#37474F", hover_color="#263238", height=40
        )
        self.btn_exit.grid(row=9, column=0, padx=20, pady=20, sticky="ew")
        
        # Main Workspace Container Frame
        self.workspace_frame = ctkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.workspace_frame.grid(row=1, column=1, sticky="nsew", padx=15, pady=15)
        self.workspace_frame.grid_rowconfigure(0, weight=1)
        self.workspace_frame.grid_columnconfigure(0, weight=1)
        
        # Build individual page frames
        self.create_dashboard_frame()
        self.create_settings_frame()
        self.create_stats_frame()
        
        # Default starting view
        self.show_dashboard_page()

    # =========================================================================
    # PAGES CREATION
    # =========================================================================
    
    def create_dashboard_frame(self):
        """Creates the camera feed & status monitoring page."""
        self.dashboard_frame = ctkinter.CTkFrame(self.workspace_frame, fg_color="transparent")
        self.dashboard_frame.grid_rowconfigure(0, weight=1)
        self.dashboard_frame.grid_columnconfigure(0, weight=3) # Camera feed card weight
        self.dashboard_frame.grid_columnconfigure(1, weight=1) # Status panel card weight
        
        # 1. Left Column: Camera Preview Box
        self.cam_card = ctkinter.CTkFrame(self.dashboard_frame, corner_radius=15, border_width=4, border_color="#333333")
        self.cam_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        self.cam_card.grid_rowconfigure(0, weight=1)
        self.cam_card.grid_columnconfigure(0, weight=1)
        
        self.video_label = ctkinter.CTkLabel(
            self.cam_card, 
            text="Camera Stream Inactive\n\nPlease load model, then press Start Camera.", 
            font=ctinker.CTkFont(family="Helvetica", size=16),
            text_color="#888888"
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
        # Bottom controls below camera inside cam_card
        self.cam_controls = ctkinter.CTkFrame(self.cam_card, height=50, fg_color="transparent")
        self.cam_controls.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        self.cam_controls.grid_columnconfigure((0, 1, 2), weight=1)
        
        self.btn_manual_screenshot = ctkinter.CTkButton(
            self.cam_controls, text="Capture Screenshot", command=self.take_manual_screenshot,
            height=32, fg_color="#455A64", hover_color="#37474F"
        )
        self.btn_manual_screenshot.grid(row=0, column=0, padx=5, sticky="ew")
        
        self.btn_fullscreen = ctkinter.CTkButton(
            self.cam_controls, text="Fullscreen Mode", command=self.toggle_fullscreen,
            height=32, fg_color="#455A64", hover_color="#37474F"
        )
        self.btn_fullscreen.grid(row=0, column=1, padx=5, sticky="ew")
        
        self.btn_clear_logs = ctkinter.CTkButton(
            self.cam_controls, text="Export CSV Logs", command=self.export_logs_dialog,
            height=32, fg_color="#455A64", hover_color="#37474F"
        )
        self.btn_clear_logs.grid(row=0, column=2, padx=5, sticky="ew")
        
        # 2. Right Column: Status Panel
        self.status_card = ctkinter.CTkFrame(self.dashboard_frame, corner_radius=15, fg_color="#1E1E1E")
        self.status_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        self.status_card.grid_columnconfigure(0, weight=1)
        
        lbl_status_header = ctkinter.CTkLabel(
            self.status_card, text="SYSTEM STATUS", 
            font=ctinker.CTkFont(family="Helvetica", size=14, weight="bold"),
            text_color="#888888"
        )
        lbl_status_header.pack(pady=(20, 15))
        
        # Status Details (Modern clean badges)
        self.create_status_badge("Time", "00:00:00")
        self.create_status_badge("Model Loaded", "False", text_color="#F44336")
        self.create_status_badge("Camera Status", "Stopped", text_color="#FFA000")
        self.create_status_badge("FPS", "0.0")
        self.create_status_badge("Consecutive Drowsy", "0 / 15")
        
        # Large Prediction Panel
        pred_label_title = ctkinter.CTkLabel(
            self.status_card, text="PREDICTION", 
            font=ctinker.CTkFont(family="Helvetica", size=11, weight="bold"),
            text_color="#9E9E9E"
        )
        pred_label_title.pack(pady=(20, 2))
        
        self.lbl_prediction = ctkinter.CTkLabel(
            self.status_card, text="INACTIVE", 
            font=ctinker.CTkFont(family="Helvetica", size=22, weight="bold"),
            text_color="#9E9E9E",
            fg_color="#2A2A2A",
            corner_radius=8,
            height=40,
            width=200
        )
        self.lbl_prediction.pack(pady=(2, 10))
        
        # Confidence progress bar
        self.lbl_confidence = ctkinter.CTkLabel(
            self.status_card, text="Confidence: 0.0%", 
            font=ctinker.CTkFont(family="Helvetica", size=12),
            text_color="#CCCCCC"
        )
        self.lbl_confidence.pack(pady=(5, 2))
        
        self.confidence_bar = ctkinter.CTkProgressBar(self.status_card, width=200, height=8)
        self.confidence_bar.set(0)
        self.confidence_bar.pack(pady=(2, 15))
        
        # Big Stop Alarm Overlay Frame (displays inside dashboard when alarm is active)
        self.alarm_panel = ctkinter.CTkFrame(self.status_card, fg_color="#C62828", corner_radius=10)
        self.lbl_alarm_warning = ctkinter.CTkLabel(
            self.alarm_panel, text="⚠ DROWSINESS DETECTED ⚠", 
            font=ctinker.CTkFont(family="Helvetica", size=14, weight="bold"),
            text_color="#FFFFFF"
        )
        self.lbl_alarm_warning.pack(pady=10)
        
        self.btn_stop_alarm = ctkinter.CTkButton(
            self.alarm_panel, text="STOP ALARM", command=self.stop_alarm,
            fg_color="#FFFFFF", hover_color="#EEEEEE", text_color="#C62828",
            font=ctinker.CTkFont(family="Helvetica", size=16, weight="bold"),
            height=40
        )
        self.btn_stop_alarm.pack(pady=(5, 10), padx=15, fill="x")

    def create_status_badge(self, label_name, initial_val, text_color=None):
        """Helper to create matching status grid indicators."""
        frame = ctkinter.CTkFrame(self.status_card, height=35, fg_color="transparent")
        frame.pack(fill="x", padx=15, pady=4)
        
        lbl_name = ctkinter.CTkLabel(
            frame, text=label_name, 
            font=ctinker.CTkFont(family="Helvetica", size=12),
            text_color="#B2B2B2"
        )
        lbl_name.pack(side="left", padx=5)
        
        lbl_val = ctkinter.CTkLabel(
            frame, text=initial_val, 
            font=ctinker.CTkFont(family="Helvetica", size=12, weight="bold")
        )
        if text_color:
            lbl_val.configure(text_color=text_color)
        lbl_val.pack(side="right", padx=5)
        
        # Save reference
        attr_name = "lbl_val_" + label_name.lower().replace(" ", "_")
        setattr(self, attr_name, lbl_val)

    def create_settings_frame(self):
        """Creates the settings form frame."""
        self.settings_frame = ctkinter.CTkFrame(self.workspace_frame, corner_radius=15, fg_color="#1E1E1E")
        self.settings_frame.grid_columnconfigure(0, weight=1)
        
        lbl_title = ctkinter.CTkLabel(
            self.settings_frame, text="Application Settings",
            font=ctinker.CTkFont(family="Helvetica", size=20, weight="bold")
        )
        lbl_title.pack(pady=(30, 20))
        
        # Form Container
        form_frame = ctkinter.CTkFrame(self.settings_frame, fg_color="transparent")
        form_frame.pack(padx=50, pady=10, fill="both", expand=True)
        form_frame.grid_columnconfigure(1, weight=1)
        
        # 1. Camera Index
        lbl_cam = ctkinter.CTkLabel(form_frame, text="Camera Index / Device ID:", font=ctinker.CTkFont(size=13))
        lbl_cam.grid(row=0, column=0, padx=20, pady=15, sticky="w")
        self.entry_cam_index = ctkinter.CTkEntry(form_frame, width=150)
        self.entry_cam_index.grid(row=0, column=1, padx=20, pady=15, sticky="w")
        
        # 2. Confidence Threshold
        lbl_conf = ctkinter.CTkLabel(form_frame, text="Drowsiness Confidence Threshold:", font=ctinker.CTkFont(size=13))
        lbl_conf.grid(row=1, column=0, padx=20, pady=15, sticky="w")
        
        conf_slider_frame = ctkinter.CTkFrame(form_frame, fg_color="transparent")
        conf_slider_frame.grid(row=1, column=1, padx=20, pady=15, sticky="ew")
        
        self.slider_confidence = ctkinter.CTkSlider(conf_slider_frame, from_=0.1, to=0.9, number_of_steps=16, command=self.update_conf_label)
        self.slider_confidence.pack(side="left", fill="x", expand=True)
        
        self.lbl_conf_val = ctkinter.CTkLabel(conf_slider_frame, text="0.50", width=40, font=ctinker.CTkFont(weight="bold"))
        self.lbl_conf_val.pack(side="right", padx=(10, 0))
        
        # 3. Consecutive Frames
        lbl_frames = ctkinter.CTkLabel(form_frame, text="Consecutive Alarm Trigger Frames:", font=ctinker.CTkFont(size=13))
        lbl_frames.grid(row=2, column=0, padx=20, pady=15, sticky="w")
        self.entry_frames = ctkinter.CTkEntry(form_frame, width=150)
        self.entry_frames.grid(row=2, column=1, padx=20, pady=15, sticky="w")
        
        # 4. Alarm Volume
        lbl_vol = ctkinter.CTkLabel(form_frame, text="Alarm Volume Level:", font=ctinker.CTkFont(size=13))
        lbl_vol.grid(row=3, column=0, padx=20, pady=15, sticky="w")
        
        vol_slider_frame = ctkinter.CTkFrame(form_frame, fg_color="transparent")
        vol_slider_frame.grid(row=3, column=1, padx=20, pady=15, sticky="ew")
        
        self.slider_volume = ctkinter.CTkSlider(vol_slider_frame, from_=0.0, to=1.0, command=self.update_vol_label)
        self.slider_volume.pack(side="left", fill="x", expand=True)
        
        self.lbl_vol_val = ctkinter.CTkLabel(vol_slider_frame, text="80%", width=40, font=ctinker.CTkFont(weight="bold"))
        self.lbl_vol_val.pack(side="right", padx=(10, 0))
        
        # 5. Theme
        lbl_theme = ctkinter.CTkLabel(form_frame, text="UI Color Theme Mode:", font=ctinker.CTkFont(size=13))
        lbl_theme.grid(row=4, column=0, padx=20, pady=15, sticky="w")
        self.opt_theme = ctkinter.CTkOptionMenu(form_frame, values=["Dark", "Light"], command=self.change_ui_theme)
        self.opt_theme.grid(row=4, column=1, padx=20, pady=15, sticky="w")
        
        # Save settings button
        btn_save = ctkinter.CTkButton(
            self.settings_frame, text="Save Settings", command=self.save_settings,
            fg_color="#2E7D32", hover_color="#1B5E20", height=40, width=200
        )
        btn_save.pack(pady=(20, 40))
        
        # Populate initially
        self.load_settings_to_ui()

    def update_conf_label(self, val):
        self.lbl_conf_val.configure(text=f"{val:.2f}")

    def update_vol_label(self, val):
        self.lbl_vol_val.configure(text=f"{int(val * 100)}%")

    def change_ui_theme(self, theme_val):
        ctinker.set_appearance_mode(theme_val)

    def load_settings_to_ui(self):
        """Fills form widgets with current config values."""
        self.entry_cam_index.delete(0, tk.END)
        self.entry_cam_index.insert(0, str(self.config.camera_index))
        
        self.slider_confidence.set(self.config.confidence_threshold)
        self.update_conf_label(self.config.confidence_threshold)
        
        self.entry_frames.delete(0, tk.END)
        self.entry_frames.insert(0, str(self.config.consecutive_frames))
        
        self.slider_volume.set(self.config.alarm_volume)
        self.update_vol_label(self.config.alarm_volume)
        
        self.opt_theme.set(self.config.theme)
        self.change_ui_theme(self.config.theme)

    def save_settings(self):
        """Validates settings from inputs and saves to configuration file."""
        try:
            cam_idx = int(self.entry_cam_index.get().strip())
            conf_t = float(self.slider_confidence.get())
            cons_f = int(self.entry_frames.get().strip())
            vol = float(self.slider_volume.get())
            theme_mode = self.opt_theme.get()
            
            if cam_idx < 0:
                raise ValueError("Camera index cannot be negative.")
            if cons_f <= 0:
                raise ValueError("Consecutive frames must be greater than 0.")
                
            # Update config and save
            self.config.camera_index = cam_idx
            self.config.confidence_threshold = conf_t
            self.config.consecutive_frames = cons_f
            self.config.alarm_volume = vol
            self.config.theme = theme_mode
            self.config.save()
            
            # Apply immediate runtime updates
            self.alarm_player.set_volume(vol)
            if self.webcam and self.webcam.camera_index != cam_idx:
                self.webcam.change_camera(cam_idx)
                
            messagebox.showinfo("Settings Saved", "Application configuration updated successfully!")
            self.show_dashboard_page()
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check settings values:\n{e}")
        except Exception as ex:
            messagebox.showerror("Error", f"Failed to save settings: {ex}")

    def create_stats_frame(self):
        """Creates the session statistics frame."""
        self.stats_frame = ctkinter.CTkFrame(self.workspace_frame, corner_radius=15, fg_color="#1E1E1E")
        self.stats_frame.grid_columnconfigure((0, 1), weight=1)
        
        lbl_title = ctkinter.CTkLabel(
            self.stats_frame, text="Monitoring Performance Statistics",
            font=ctinker.CTkFont(family="Helvetica", size=20, weight="bold")
        )
        lbl_title.grid(row=0, column=0, columnspan=2, pady=(30, 20))
        
        # Create metric boxes
        self.stat_total_frames_lbl = self.create_stat_metric("Total Frames Processed", "0", 1, 0)
        self.stat_drowsy_frames_lbl = self.create_stat_metric("Total Drowsy Detections", "0", 1, 1)
        self.stat_total_alarms_lbl = self.create_stat_metric("Total Alarms Triggered", "0", 2, 0)
        self.stat_running_time_lbl = self.create_stat_metric("Active Session Runtime", "00h 00m 00s", 2, 1)
        self.stat_today_detections_lbl = self.create_stat_metric("Today's Detections", "0", 3, 0)
        
        # Empty placeholder spacer in column 1 row 3
        dummy = ctkinter.CTkFrame(self.stats_frame, fg_color="transparent")
        dummy.grid(row=3, column=1, pady=15, padx=25, sticky="nsew")
        
        # Export logs button inside stats page
        btn_export = ctkinter.CTkButton(
            self.stats_frame, text="Export Logs (CSV)", command=self.export_logs_dialog,
            fg_color="#1A237E", hover_color="#283593", height=40, width=220
        )
        btn_export.grid(row=4, column=0, columnspan=2, pady=(30, 40))

    def create_stat_metric(self, title, initial_val, r, c):
        card = ctkinter.CTkFrame(self.stats_frame, corner_radius=10, fg_color="#2A2A2A", height=100)
        card.grid(row=r, column=c, padx=25, pady=15, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        
        lbl_title = ctkinter.CTkLabel(
            card, text=title, 
            font=ctinker.CTkFont(family="Helvetica", size=13, weight="bold"),
            text_color="#9E9E9E"
        )
        lbl_title.pack(pady=(15, 5))
        
        lbl_val = ctkinter.CTkLabel(
            card, text=initial_val, 
            font=ctinker.CTkFont(family="Helvetica", size=24, weight="bold"),
            text_color="#FFFFFF"
        )
        lbl_val.pack(pady=(0, 15))
        return lbl_val

    # =========================================================================
    # NAVIGATION LOGIC
    # =========================================================================
    
    def show_dashboard_page(self):
        self.settings_frame.grid_forget()
        self.stats_frame.grid_forget()
        self.dashboard_frame.grid(row=0, column=0, sticky="nsew")
        
        self.btn_dashboard.configure(fg_color="#1A237E")
        self.btn_settings.configure(fg_color="#3B3B3B")
        self.btn_stats.configure(fg_color="#3B3B3B")

    def show_settings_page(self):
        self.dashboard_frame.grid_forget()
        self.stats_frame.grid_forget()
        self.settings_frame.grid(row=0, column=0, sticky="nsew")
        
        self.btn_dashboard.configure(fg_color="#3B3B3B")
        self.btn_settings.configure(fg_color="#1A237E")
        self.btn_stats.configure(fg_color="#3B3B3B")
        self.load_settings_to_ui()

    def show_stats_page(self):
        self.dashboard_frame.grid_forget()
        self.settings_frame.grid_forget()
        self.stats_frame.grid(row=0, column=0, sticky="nsew")
        
        self.btn_dashboard.configure(fg_color="#3B3B3B")
        self.btn_settings.configure(fg_color="#3B3B3B")
        self.btn_stats.configure(fg_color="#1A237E")
        self.update_statistics_ui()

    def update_statistics_ui(self):
        """Refreshes values on stats cards."""
        self.stat_total_frames_lbl.configure(text=str(self.stat_total_frames))
        self.stat_drowsy_frames_lbl.configure(text=str(self.stat_total_drowsy_frames))
        self.stat_total_alarms_lbl.configure(text=str(self.stat_total_alarms))
        self.stat_today_detections_lbl.configure(text=str(self.stat_today_detections))
        
        if self.session_start_time:
            runtime = int(time.time() - self.session_start_time)
            h = runtime // 3600
            m = (runtime % 3600) // 60
            s = runtime % 60
            self.stat_running_time_lbl.configure(text=f"{h:02d}h {m:02d}m {s:02d}s")
        else:
            self.stat_running_time_lbl.configure(text="00h 00m 00s")

    # =========================================================================
    # SPLASH / LOADING SEQUENCE
    # =========================================================================
    
    def show_splash_screen(self):
        """Creates a smooth loading animation card over the dashboard."""
        self.splash_overlay = ctkinter.CTkFrame(self.workspace_frame, fg_color="#121212")
        self.splash_overlay.grid(row=0, column=0, sticky="nsew")
        
        self.splash_overlay.grid_rowconfigure((0, 1, 2, 3), weight=1)
        self.splash_overlay.grid_columnconfigure(0, weight=1)
        
        lbl_splash_title = ctkinter.CTkLabel(
            self.splash_overlay, text="DDD SYSTEM INITIALIZATION",
            font=ctinker.CTkFont(family="Helvetica", size=24, weight="bold"),
            text_color="#FFFFFF"
        )
        lbl_splash_title.grid(row=1, column=0, sticky="s", pady=20)
        
        self.lbl_splash_status = ctkinter.CTkLabel(
            self.splash_overlay, text="Checking configuration files...",
            font=ctinker.CTkFont(family="Helvetica", size=14),
            text_color="#888888"
        )
        self.lbl_splash_status.grid(row=2, column=0, sticky="n", pady=5)
        
        self.splash_progress = ctkinter.CTkProgressBar(self.splash_overlay, width=350, height=8)
        self.splash_progress.grid(row=2, column=0, pady=(40, 0))
        self.splash_progress.set(0)
        
        # Start loading sequence in a background thread
        threading.Thread(target=self._run_loading_sequence, daemon=True).start()

    def _run_loading_sequence(self):
        """Simulates background modules startup checkpoints."""
        steps = [
            ("Initializing GUI subsystems...", 0.2),
            ("Loading Pygame mixer and audio tracks...", 0.4),
            ("Instantiating MediaPipe Face Detection...", 0.6),
            ("Attempting to load Keras model (TF/MobileNet)...", 0.8),
            ("Verification complete. Application ready.", 1.0)
        ]
        
        for status, progress in steps:
            time.sleep(0.4)
            self._update_splash_progress(status, progress)
            
            # Load model step
            if progress == 0.8:
                # Load model
                success, msg = self.model_loader.load_model(self.config.model_path)
                # If loaded successfully, update labels
                self.after(0, lambda m=msg, s=success: self._handle_model_loaded(s, m))
                
        # Fade out splash
        time.sleep(0.5)
        self.after(0, self.splash_overlay.destroy)

    def _update_splash_progress(self, status_text, progress_val):
        self.lbl_splash_status.configure(text=status_text)
        self.splash_progress.set(progress_val)

    def _handle_model_loaded(self, success, message):
        """Callback to handle model loading outcomes."""
        if success:
            self.lbl_val_model_loaded.configure(text="Loaded", text_color="#4CAF50")
            self.btn_start_cam.configure(state="normal")
            print(message)
        else:
            self.lbl_val_model_loaded.configure(text="Demo Mode", text_color="#FFA000")
            # We enable Start Camera anyway since we support DEMO simulation mode
            self.btn_start_cam.configure(state="normal")
            print(message)

    # =========================================================================
    # MODEL LOADER DIALOG
    # =========================================================================
    
    def load_model_dialog(self):
        """Prompts the user to load a custom .keras file."""
        file_path = filedialog.askopenfilename(
            title="Load TensorFlow Keras Model",
            filetypes=[("Keras model files", "*.keras"), ("All files", "*.*")]
        )
        if not file_path:
            return
            
        # Stop webcam if it is running
        webcam_running = self.camera_running
        if webcam_running:
            self.stop_camera()
            
        # Show loading dialog
        loading_popup = tk.Toplevel(self)
        loading_popup.title("Loading Model")
        loading_popup.geometry("300x120")
        loading_popup.resizable(False, False)
        # Center popup relative to main window
        x = self.winfo_x() + (self.winfo_width() // 2) - 150
        y = self.winfo_y() + (self.winfo_height() // 2) - 60
        loading_popup.geometry(f"+{x}+{y}")
        loading_popup.configure(bg="#1E1E1E")
        
        lbl = ctkinter.CTkLabel(
            loading_popup, text="Importing Keras weights...\nPlease wait.",
            font=ctinker.CTkFont(family="Helvetica", size=13),
            text_color="#FFFFFF"
        )
        lbl.pack(pady=20)
        
        pb = ctkinter.CTkProgressBar(loading_popup, width=200)
        pb.pack()
        pb.start()
        
        def _load_model_thread():
            success, msg = self.model_loader.load_model(file_path)
            
            def _done():
                loading_popup.destroy()
                if success:
                    self.lbl_val_model_loaded.configure(text="Loaded", text_color="#4CAF50")
                    self.btn_start_cam.configure(state="normal")
                    self.predictor.reset_alarm_state()
                    messagebox.showinfo("Model Loaded", "Model Loaded Successfully.")
                    # Persist the new model path
                    self.config.model_path = file_path
                    self.config.save()
                else:
                    self.lbl_val_model_loaded.configure(text="Demo Mode", text_color="#FFA000")
                    messagebox.showwarning("Loading Fallback", msg)
                
                # Resume camera if it was running
                if webcam_running:
                    self.start_camera()
                    
            self.after(0, _done)
            
        threading.Thread(target=_load_model_thread, daemon=True).start()

    # =========================================================================
    # CAMERA WORKFLOW
    # =========================================================================
    
    def start_camera(self):
        """Starts the multi-threaded camera stream and inference loop."""
        if self.camera_running:
            return
            
        self.webcam = WebcamStream(camera_index=self.config.camera_index)
        self.webcam.start()
        
        self.camera_running = True
        self.session_start_time = time.time()
        
        self.btn_start_cam.configure(state="disabled")
        self.btn_stop_cam.configure(state="normal")
        self.lbl_val_camera_status.configure(text="Connected", text_color="#4CAF50")
        
        # Hide alarm panel initially
        self.alarm_panel.pack_forget()
        
        # Reset predictor
        self.predictor.reset_alarm_state()
        
        # Begin frame update cycle
        self.update_frame_loop()

    def stop_camera(self):
        """Terminates camera stream and updates GUI indicators."""
        if not self.camera_running:
            return
            
        self.camera_running = False
        if self.webcam:
            self.webcam.stop()
            self.webcam = None
            
        self.btn_start_cam.configure(state="normal")
        self.btn_stop_cam.configure(state="disabled")
        self.lbl_val_camera_status.configure(text="Stopped", text_color="#FFA000")
        self.lbl_val_fps.configure(text="0.0")
        self.lbl_prediction.configure(text="INACTIVE", text_color="#9E9E9E", fg_color="#2A2A2A")
        self.lbl_confidence.configure(text="Confidence: 0.0%")
        self.confidence_bar.set(0)
        
        self.video_label.configure(
            image=None, 
            text="Camera Stream Stopped.\n\nPress Start Camera to resume."
        )
        self.cam_card.configure(border_color="#333333")
        
        # Stop alarm sound if running
        self.stop_alarm()

    def reconnect_camera(self):
        """Forces stop and restart of camera stream."""
        if self.camera_running:
            self.stop_camera()
            self.after(500, self.start_camera)
        else:
            self.start_camera()

    def update_frame_loop(self):
        """Main periodic GUI loop (approx 30 FPS) for drawing frame and processing predictions."""
        if not self.camera_running or self.webcam is None:
            return
            
        frame, fps = self.webcam.read()
        
        if frame is not None:
            self.stat_total_frames += 1
            h, w, c = frame.shape
            
            # 1. Run MediaPipe Face Detection
            face_crop, bbox, annotated_frame = self.face_detector.detect_and_crop(frame)
            
            # Default outputs
            prediction_label = "No Face"
            confidence_val = 0.0
            consecutive_drowsy_str = "0 / " + str(self.config.consecutive_frames)
            
            if face_crop is not None:
                # Draw green bounding box around face
                if bbox:
                    xmin, ymin, width, height = bbox
                    # BGR colors: (B, G, R)
                    # We will color the box green if not drowsy, red if alarm is active
                    box_color = (0, 0, 255) if self.alarm_player.is_playing else (0, 255, 0)
                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmin+width, ymin+height), box_color, 3)
                
                # 2. Run prediction & smoothing
                label, conf, count, should_alarm = self.predictor.predict_and_smooth(face_crop)
                
                prediction_label = label
                confidence_val = conf
                consecutive_drowsy_str = f"{count} / {self.config.consecutive_frames}"
                
                if label == "Drowsy":
                    self.stat_total_drowsy_frames += 1
                    # Track daily detections simple increment
                    self.stat_today_detections += 1
                
                # Check alarm trigger condition
                if should_alarm and not self.alarm_player.is_playing:
                    self.trigger_alarm(annotated_frame, label, conf)
            else:
                # No face detected
                cv2.putText(
                    annotated_frame, "No Face Detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
                )
                # If we lose face during high-consecutive frames, count doesn't reset immediately,
                # but we will display warning message in GUI
                prediction_label = "No Face Detected"
                confidence_val = 0.0
                # Reset consecutive count if face is lost and alarm not active
                if not self.predictor.alarm_active:
                    self.predictor.consecutive_drowsy_count = 0
                consecutive_drowsy_str = f"{self.predictor.consecutive_drowsy_count} / {self.config.consecutive_frames}"

            # 3. Update GUI Stats Panel Panel values
            self.lbl_val_fps.configure(text=f"{fps:.1f}")
            self.lbl_val_consecutive_drowsy.configure(text=consecutive_drowsy_str)
            self.lbl_val_time.configure(text=datetime.now().strftime("%H:%M:%S"))
            
            # Prediction Label Styling
            if prediction_label == "Drowsy":
                self.lbl_prediction.configure(text="DROWSY", text_color="#FFFFFF", fg_color="#C62828")
                self.cam_card.configure(border_color="#C62828")
            elif prediction_label == "Non Drowsy":
                self.lbl_prediction.configure(text="ACTIVE", text_color="#FFFFFF", fg_color="#2E7D32")
                self.cam_card.configure(border_color="#2E7D32")
            else:
                self.lbl_prediction.configure(text="NO FACE DETECTED", text_color="#FFFFFF", fg_color="#FFA000")
                self.cam_card.configure(border_color="#FFA000")
                
            self.lbl_confidence.configure(text=f"Confidence: {confidence_val * 100:.1f}%")
            self.confidence_bar.set(confidence_val)
            
            # Handle Alarm Flashing visual effect
            if self.alarm_player.is_playing:
                self.flash_state = not self.flash_state
                flash_color = "#C62828" if self.flash_state else "#121212"
                self.cam_card.configure(border_color="#C62828" if self.flash_state else "#D32F2F")
                self.lbl_val_alarm_status.configure(text="ACTIVE", text_color="#F44336")
                # Flash header or window background slightly (to look premium)
                # But to avoid heavy drawing lag we just toggle card outline/panel colors
                self.alarm_panel.configure(fg_color="#C62828" if self.flash_state else "#B71C1C")
            else:
                self.lbl_val_alarm_status.configure(text="Inactive", text_color="#4CAF50")
            
            # 4. Render Frame in TK Canvas/Label
            # MediaPipe/OpenCV uses BGR, convert to RGB for PIL
            rgb_img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Resize image to fit cam_card dimensions dynamically
            card_w = self.cam_card.winfo_width() - 10
            card_h = self.cam_card.winfo_height() - 80 # Leave space for buttons
            
            if card_w > 100 and card_h > 100:
                # Maintain aspect ratio
                h_img, w_img, _ = rgb_img.shape
                ratio = min(card_w / w_img, card_h / h_img)
                new_w = int(w_img * ratio)
                new_h = int(h_img * ratio)
                
                rgb_resized = cv2.resize(rgb_img, (new_w, new_h))
                pil_image = Image.fromarray(rgb_resized)
                
                # Map to CustomTkinter CTkImage to scale automatically
                ctk_img = ctkinter.CTkImage(light_image=pil_image, dark_image=pil_image, size=(new_w, new_h))
                
                self.video_label.configure(image=ctk_img, text="")
                
        else:
            # Camera disconnect check
            if not self.webcam.is_connected:
                self.lbl_val_camera_status.configure(text="Disconnected", text_color="#F44336")
                self.video_label.configure(
                    image=None, 
                    text="CAMERA DISCONNECTED!\n\nPlease check connection and press Reconnect Cam."
                )
                self.cam_card.configure(border_color="#F44336")
                
        # Repeat every 30 milliseconds (approx 33 FPS)
        self.after(30, self.update_frame_loop)

    # =========================================================================
    # ALARM MANAGEMENT & TRIGGERS
    # =========================================================================
    
    def trigger_alarm(self, current_frame, label, confidence):
        """Fires Pygame alarm, saves alert logs and screenshot, and shows Alert UI."""
        self.alarm_player.play()
        self.stat_total_alarms += 1
        self.alarm_start_time = time.time()
        
        # Show Alarm Action Panel
        self.alarm_panel.pack(pady=10, fill="x", padx=15)
        
        # Save screenshot asynchronously to not block the UI thread
        threading.Thread(
            target=save_screenshot, 
            args=(current_frame, self.config.screenshots_dir), 
            daemon=True
        ).start()
        
        # Write log entry
        log_prediction(self.config.logs_dir, label, confidence)

    def stop_alarm(self):
        """Halts the alarm player and resets counters."""
        self.alarm_player.stop()
        self.predictor.reset_alarm_state()
        self.alarm_panel.pack_forget()
        
        # Restore colors
        self.cam_card.configure(border_color="#333333")
        self.lbl_val_consecutive_drowsy.configure(text=f"0 / {self.config.consecutive_frames}")

    # =========================================================================
    # ADDITIONAL EXTRA APP FEATURES
    # =========================================================================
    
    def take_manual_screenshot(self):
        """Allows taking a screenshot on request."""
        if not self.camera_running or self.webcam is None:
            messagebox.showwarning("Action Blocked", "Please start camera to take a screenshot.")
            return
            
        frame, _ = self.webcam.read()
        if frame is not None:
            filepath = save_screenshot(frame, self.config.screenshots_dir, filename_prefix="manual")
            if filepath:
                messagebox.showinfo("Screenshot Saved", f"Saved successfully to:\n{filepath}")
            else:
                messagebox.showerror("Error", "Failed to save screenshot.")

    def export_logs_dialog(self):
        """Opens file dialog for exporting the CSV log."""
        log_file = os.path.join(self.config.logs_dir, "drowsiness_log.csv")
        if not os.path.exists(log_file):
            messagebox.showwarning("Export Empty", "No data logs are available to export yet.")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Export CSV Log File",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile="drowsiness_log_export.csv"
        )
        if not save_path:
            return
            
        try:
            export_logs(self.config.logs_dir, save_path)
            messagebox.showinfo("Export Success", f"Logs exported successfully to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Could not export logs: {e}")

    def toggle_fullscreen(self):
        """Toggles window fullscreen mode."""
        self.is_fullscreen = not self.is_fullscreen
        self.attributes("-fullscreen", self.is_fullscreen)
        if self.is_fullscreen:
            self.btn_fullscreen.configure(text="Exit Fullscreen")
        else:
            self.btn_fullscreen.configure(text="Fullscreen Mode")

    def on_exit(self):
        """Safely stops threads and exits."""
        if messagebox.askokcancel("Exit DDD", "Are you sure you want to exit Driver Drowsiness Detection?"):
            self.stop_camera()
            self.alarm_player.stop()
            pygame.quit()
            self.destroy()
