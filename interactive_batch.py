import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import shutil

def reset_folders():
    folders = ['hasil_analisis', 'hasil_histogram']
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

def main():
    reset_folders()
    
    input_folder = 'dataset_beras' 
    image_paths = glob.glob(f"{input_folder}/*.jpg") + glob.glob(f"{input_folder}/*.png")
    
    if len(image_paths) == 0:
        print(f"[ERROR] Tidak ada gambar di folder '{input_folder}'.")
        return

    print(f"[INFO] Memulai Interactive Batch untuk {len(image_paths)} gambar...")
    
    all_raw_areas = [] 
    total_rice_global = 0
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for index, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None: continue
            
        print(f"\n--- Memproses {index+1}/{len(image_paths)}: {filename} ---")
        roi_rect = cv2.selectROI(f"Seleksi ROI - {filename}", image, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(f"Seleksi ROI - {filename}")
        
        x, y, w, h = roi_rect
        if w == 0 or h == 0: continue

        roi_image = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        thresh_val = 127
        local_areas = []
        valid_count = 0
        final_output = None

        while True:
            _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
            morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_ellipse, iterations=1)
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            output_img = roi_image.copy()
            valid_count = 0
            temp_areas = []
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 20: 
                    valid_count += 1
                    temp_areas.append(area)
                    cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
            
            cv2.rectangle(output_img, (5, 5), (300, 50), (0, 0, 0), -1)
            cv2.putText(output_img, f"File: {filename}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(output_img, f"Thresh: {thresh_val} | Butir: {valid_count}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow(f"Analisis - {filename}", output_img)
            
            key = cv2.waitKey(0) & 0xFF
            if key in [ord('w'), 82]: thresh_val = min(255, thresh_val + 2)
            elif key in [ord('s'), 84]: thresh_val = max(0, thresh_val - 2)
            elif key in [13, ord('q')]: 
                local_areas = temp_areas 
                final_output = output_img # Simpan frame terakhir saat ENTER ditekan
                cv2.destroyWindow(f"Analisis - {filename}")
                break

        # SIMPAN GAMBAR HASIL KE FOLDER
        if final_output is not None:
            cv2.imwrite(os.path.join('hasil_analisis', f"hybrid_{filename}"), final_output)

        all_raw_areas.extend(local_areas)
        total_rice_global += valid_count
        print(f"[SUCCESS] Disimpan: {valid_count} butir dari {filename}.")

    print(f"\n[SELESAI] Total beras dari {len(image_paths)} foto: {total_rice_global} butir.")

    # SIMPAN DAN TAMPILKAN HISTOGRAM GLOBAL
    if len(all_raw_areas) > 0:
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 6))
        plt.hist(all_raw_areas, bins=25, color='darkorange', edgecolor='black')
        plt.axvline(np.mean(all_raw_areas), color='blue', linestyle='dashed', linewidth=2, label=f'Rata-rata: {np.mean(all_raw_areas):.1f}')
        plt.axvline(np.median(all_raw_areas), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(all_raw_areas):.1f}')
        plt.title(f'Histogram Distribusi Hybrid Batch ({total_rice_global} Butir)')
        plt.xlabel('Luas Area Beras (Piksel Persegi)')
        plt.ylabel('Frekuensi')
        plt.legend()
        
        # Simpan grafik
        plt.savefig(os.path.join('hasil_histogram', 'histogram_hybrid_batch.png'))
        print("[SUCCESS] Grafik disimpan di folder 'hasil_histogram'.")
        plt.show()

if __name__ == "__main__":
    main()