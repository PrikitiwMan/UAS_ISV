import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def reset_folders():
    folders = ['hasil_analisis', 'hasil_histogram']
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder) # Hapus folder beserta isinya
        os.makedirs(folder)       # Buat folder baru yang kosong

def main():
    reset_folders()
    
    image_path = 'beras.jpg' 
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"[ERROR] Citra {image_path} tidak ditemukan.")
        return

    print("[INFO] Seleksi ROI area beras menggunakan mouse, lalu tekan ENTER.")
    roi_rect = cv2.selectROI("1. Seleksi Area", image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("1. Seleksi Area")
    
    x, y, w, h = roi_rect
    if w == 0 or h == 0: return

    roi_image = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh_val = 127
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    final_output = None
    final_areas = []

    print("[INFO] KEYBOARD: [W] Naik Thresh, [S] Turun, [ENTER] Selesai & Simpan.")
    while True:
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_ellipse, iterations=1)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output_img = roi_image.copy()
        raw_areas = []
        valid_contours = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 20:
                raw_areas.append(area)
                valid_contours.append(cnt)
                
        rice_count = len(valid_contours)
        
        if rice_count > 0:
            p33, p66 = np.percentile(raw_areas, 33), np.percentile(raw_areas, 66)
            stats = {"Patah": 0, "Normal": 0, "Utuh": 0}
            
            for i, cnt in enumerate(valid_contours):
                area = raw_areas[i]
                if area <= p33: color, stats["Patah"] = (0, 0, 255), stats["Patah"]+1
                elif area <= p66: color, stats["Normal"] = (0, 255, 255), stats["Normal"]+1
                else: color, stats["Utuh"] = (0, 255, 0), stats["Utuh"]+1
                    
                cv2.drawContours(output_img, [cnt], -1, color, 2)
            
            avg_area = np.mean(raw_areas)
        else:
            avg_area, stats = 0, {"Patah": 0, "Normal": 0, "Utuh": 0}

        cv2.rectangle(output_img, (5, 5), (300, 70), (0, 0, 0), -1)
        cv2.putText(output_img, f"Thresh: {thresh_val} | Total: {rice_count}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output_img, f"Avg Area: {avg_area:.1f} px", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output_img, f"Patah:{stats['Patah']} Nrml:{stats['Normal']} Utuh:{stats['Utuh']}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        cv2.imshow("2. Analisis", output_img)
        
        key = cv2.waitKey(0) & 0xFF
        if key in [ord('w'), 82]: thresh_val = min(255, thresh_val + 2)
        elif key in [ord('s'), 84]: thresh_val = max(0, thresh_val - 2)
        elif key in [13, ord('q')]: 
            final_output = output_img
            final_areas = raw_areas
            break

    cv2.destroyAllWindows()

    # SIMPAN GAMBAR HASIL
    if final_output is not None:
        cv2.imwrite(os.path.join('hasil_analisis', 'result_tunggal.jpg'), final_output)
        print("[SUCCESS] Gambar masking disimpan di folder 'hasil_analisis'.")

    # SIMPAN DAN TAMPILKAN HISTOGRAM
    if len(final_areas) > 0:
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 6))
        plt.hist(final_areas, bins=20, color='darkorange', edgecolor='black')
        plt.axvline(np.mean(final_areas), color='blue', linestyle='dashed', linewidth=2, label=f'Rata-rata: {np.mean(final_areas):.1f}')
        plt.axvline(np.median(final_areas), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(final_areas):.1f}')
        plt.title('Histogram Distribusi Ukuran Butir Beras')
        plt.xlabel('Luas Area Beras (Piksel Persegi)')
        plt.ylabel('Frekuensi')
        plt.legend()
        
        # Simpan grafik sebelum di-show
        plt.savefig(os.path.join('hasil_histogram', 'histogram_tunggal.png'))
        print("[SUCCESS] Grafik disimpan di folder 'hasil_histogram'.")
        plt.show()

if __name__ == "__main__":
    main()