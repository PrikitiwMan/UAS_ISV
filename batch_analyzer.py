import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def main():
    # 1. PERSIAPAN FOLDER
    input_folder = 'dataset_beras'   # Folder tempat kamu menaruh banyak foto
    output_folder = 'hasil_analisis' # Folder untuk menyimpan foto hasil masking otomatis
    
    # Buat folder output otomatis jika belum ada
    os.makedirs(output_folder, exist_ok=True)
    
    # Ambil semua file dengan ekstensi .jpg dan .png di folder input
    image_paths = glob.glob(f"{input_folder}/*.jpg") + glob.glob(f"{input_folder}/*.png")
    
    if len(image_paths) == 0:
        print(f"[ERROR] Tidak ada gambar ditemukan di folder '{input_folder}'.")
        return

    print(f"[INFO] Memulai Batch Processing untuk {len(image_paths)} gambar...\n")
    
    all_raw_areas = [] # Menyimpan data ukuran beras dari SELURUH foto
    total_rice_global = 0

    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 2. PROSES LOOPING (SCAN BANYAK FOTO)
    for index, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
            
        # Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # OTSU'S THRESHOLDING (Auto Threshold)
        # Algoritma ini mencari nilai threshold terbaik secara otomatis (mengabaikan angka 0 di parameter)
        best_thresh, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morfologi
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_ellipse, iterations=1)
        
        # Find Contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output_img = image.copy()
        valid_count = 0
        local_areas = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 20: # Filter debu
                valid_count += 1
                total_rice_global += 1
                all_raw_areas.append(area)
                local_areas.append(area)
                
                # Masking hijau polos
                cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
        
        # Tulis HUD sederhana di pojok kiri atas
        avg_local = np.mean(local_areas) if len(local_areas) > 0 else 0
        cv2.rectangle(output_img, (5, 5), (350, 70), (0, 0, 0), -1)
        cv2.putText(output_img, f"File: {filename}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output_img, f"Auto-Thresh: {int(best_thresh)} | Butir: {valid_count}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Simpan hasil masking ke folder output
        save_path = os.path.join(output_folder, f"result_{filename}")
        cv2.imwrite(save_path, output_img)
        
        print(f"[{index+1}/{len(image_paths)}] Berhasil memproses {filename} -> Ditemukan {valid_count} butir beras.")

    print(f"\n[SELESAI] Total seluruh beras dari {len(image_paths)} foto: {total_rice_global} butir.")
    print(f"Gambar hasil masking telah disimpan di folder: {output_folder}/")

    # 3. ANALISIS HISTOGRAM GLOBAL
    if len(all_raw_areas) > 0:
        print("\n[INFO] Membuka grafik Histogram Distribusi Global...")
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 6))
        
        plt.hist(all_raw_areas, bins=25, color='darkorange', edgecolor='black', alpha=0.8)
        
        mean_val = np.mean(all_raw_areas)
        median_val = np.median(all_raw_areas)
        
        plt.axvline(mean_val, color='blue', linestyle='dashed', linewidth=2, label=f'Rata-rata: {mean_val:.1f} px')
        plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.1f} px')
        
        plt.title(f'Histogram Distribusi Ukuran dari Total {total_rice_global} Butir Beras')
        plt.xlabel('Luas Area Beras (Piksel Persegi)')
        plt.ylabel('Frekuensi (Jumlah Butir)')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()