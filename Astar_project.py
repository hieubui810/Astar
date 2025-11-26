import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np # Import thêm numpy để xử lý mảng tốt hơn nếu cần

# ==========================================
# 1. DỮ LIỆU BẢN ĐỒ
# ==========================================
mini_map = [
    ["1", ".", "2", "3", ".", "4", "5", ".", "6", "7"],
    [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    ["8", "9", ".", "10", "11", "12", ".", "13", "14", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    ["15", "16", ".", "17", "18", ".", "19", "20", ".", "21"],
    [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    ["22", "23", ".", "24", "25", "26", ".", "27", "28", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    ["29", "30", ".", "31", "32", "33", ".", "34", "35", "."]
]

# ==========================================
# 2. CÁC HÀM LOGIC A* (Giữ nguyên)
# ==========================================
def tim_vi_tri(giatri, luoi):
    for r, dong in enumerate(luoi):
        for c, o in enumerate(dong):
            if o == giatri: return (r, c)
    return None

def heuristic(nut, dich):
    # Khoảng cách Manhattan
    return abs(nut[0] - dich[0]) + abs(nut[1] - dich[1])

def hang_xom(nut, batdau, dich, luoi):
    dong, cot = nut
    huong = [(-1,0), (1,0), (0,-1), (0,1)]
    ket_qua = []
    for dx, dy in huong:
        nd, nc = dong + dx, cot + dy
        if 0 <= nd < len(luoi) and 0 <= nc < len(luoi[0]):
            o = luoi[nd][nc]
            # Đi được nếu là đường (.) hoặc điểm Start/End
            if o == "." or (nd, nc) == batdau or (nd, nc) == dich:
                ket_qua.append((nd, nc))
    return ket_qua

def a_sao(batdau, dich, luoi):
    mo = []; heapq.heappush(mo, (0, batdau))
    dong = set()
    g_diem = {batdau: 0}
    f_diem = {batdau: heuristic(batdau, dich)}
    cha = {batdau: None}
    so_node_duyet = 0 

    while mo:
        _, nut_hien_tai = heapq.heappop(mo)
        if nut_hien_tai in dong: continue
        dong.add(nut_hien_tai)
        so_node_duyet += 1

        if nut_hien_tai == dich:
            duong_di = []
            cur = nut_hien_tai
            while cur:
                duong_di.append(cur)
                cur = cha[cur]
            duong_di.reverse()
            return duong_di, g_diem[dich], so_node_duyet

        for xom in hang_xom(nut_hien_tai, batdau, dich, luoi):
            if xom in dong: continue
            g_tam = g_diem[nut_hien_tai] + 1
            if xom not in g_diem or g_tam < g_diem[xom]:
                g_diem[xom] = g_tam
                f_diem[xom] = g_tam + heuristic(xom, dich)
                cha[xom] = nut_hien_tai
                heapq.heappush(mo, (f_diem[xom], xom))
    return None, 0, so_node_duyet

# ==========================================
# 3. HÀM VẼ GIAO DIỆN PRO (Tích hợp số liệu & công thức)
# ==========================================
# Cập nhật thêm tham số cost và opened
def ve_do_thi_chuyen_nghiep(luoi, duong_di, batdau, dich, cost, opened, show_grid_labels=True):
    rows = len(luoi)
    cols = len(luoi[0])
    
    # Tăng kích thước khung hình để có chỗ cho phần text bên phải
    fig, ax = plt.subplots(figsize=(12, 7)) 
    
    # --- PHẦN 1: VẼ BẢN ĐỒ (Bên trái) ---
    color_map = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if luoi[r][c] != '.' and (r, c) != batdau and (r, c) != dich:
                color_map[r][c] = 1 
    
    cmap = mcolors.ListedColormap(['white', '#6FA8DC']) 
    ax.imshow(color_map, cmap=cmap, origin='upper')

    ax.set_xticks([x - 0.5 for x in range(1, cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, rows)], minor=True)
    ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    if duong_di:
        path_y = [p[0] for p in duong_di]
        path_x = [p[1] for p in duong_di]
        ax.plot(path_x, path_y, color='#F29F3F', linewidth=5, zorder=2, label='Đường đi')

    ax.text(batdau[1], batdau[0], 'A', ha='center', va='center', color='black', fontsize=16, fontweight='bold', zorder=3)
    ax.text(dich[1], dich[0], 'B', ha='center', va='center', color='black', fontsize=16, fontweight='bold', zorder=3)
            
    for r in range(rows):
        for c in range(cols):
            txt = luoi[r][c]
            if txt != '.':
                ax.text(c, r, txt, ha='center', va='center', color='black', fontsize=9) 

    if show_grid_labels:
        ax.set_xticks(range(cols)); ax.set_xticklabels([f"{i}" for i in range(cols)], fontsize=9)
        ax.set_yticks(range(rows)); ax.set_yticklabels([f"{i}" for i in range(rows)], fontsize=9)
    else:
        ax.set_xticks([]); ax.set_yticks([])

    ax.set_xlim(-0.5, cols-0.5); ax.set_ylim(rows-0.5, -0.5)
    plt.title(f"A* Pathfinding Demo: {luoi[batdau[0]][batdau[1]]} -> {luoi[dich[0]][dich[1]]}", pad=20, fontsize=14, fontweight='bold')

    # --- PHẦN 2: VẼ SỐ LIỆU & CÔNG THỨC (Bên phải) ---
    # Điều chỉnh lề phải để tạo khoảng trống cho text
    plt.subplots_adjust(right=0.7)
    
    # Vị trí x bắt đầu của cột text (tương đối so với khung hình, từ 0 đến 1)
    text_x = 0.75 
    
    # 2.1. Thống kê kết quả
    plt.figtext(text_x, 0.85, "KẾT QUẢ THỰC NGHIỆM", fontsize=12, fontweight='bold', color='darkblue')
    plt.figtext(text_x, 0.80, f"• Tổng chi phí (Cost): {cost}", fontsize=11)
    plt.figtext(text_x, 0.76, f"• Số node đã duyệt: {opened}", fontsize=11)
    
    # 2.2. Công thức thuật toán (Sử dụng LaTeX để render đẹp mắt)
    plt.figtext(text_x, 0.65, "CÔNG THỨC A* CỐT LÕI", fontsize=12, fontweight='bold', color='darkred')
    # Sử dụng raw string (r"...") để viết công thức LaTeX
    plt.figtext(text_x, 0.58, r"$f(n) = g(n) + h(n)$", fontsize=14, color='black', bbox={'facecolor':'yellow', 'alpha':0.2, 'pad':5})
    
    plt.figtext(text_x, 0.52, "Trong đó:", fontsize=10, fontstyle='italic')
    plt.figtext(text_x+0.02, 0.48, "- f(n): Tổng chi phí ước tính", fontsize=10)
    plt.figtext(text_x+0.02, 0.45, "- g(n): Chi phí thực từ Start", fontsize=10)
    plt.figtext(text_x+0.02, 0.42, "- h(n): Heuristic ước tính đến End", fontsize=10)

    # 2.3. Hàm Heuristic sử dụng
    plt.figtext(text_x, 0.32, "HEURISTIC (MANHATTAN)", fontsize=12, fontweight='bold', color='green')
    plt.figtext(text_x, 0.25, r"$h(n) = |x_1 - x_2| + |y_1 - y_2|$", fontsize=12, color='black')

    # Lưu ảnh chất lượng cao
    filename = f"astar_result_{luoi[batdau[0]][batdau[1]]}_to_{luoi[dich[0]][dich[1]]}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📸 Đã lưu ảnh kết quả chuyên nghiệp vào file: {filename}")
    plt.show()

# ==========================================
# 4. IN BẢN ĐỒ GỐC
# ==========================================
def in_ban_do_goc(luoi):
    print("\n[BẢN ĐỒ HIỆN TẠI]")
    cols = len(luoi[0])
    print("      " + "".join(f"{c:<4}" for c in range(cols)))
    print("    " + "="*(cols*4))
    for r, dong in enumerate(luoi):
        print(f"H {r:<2} |" + "".join(f"{o:>3} " for o in dong))
    print("\n")

# ==========================================
# 5. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    in_ban_do_goc(mini_map) 
    print("--- TÌM ĐƯỜNG A* PRO VISUALIZATION ---")
    b_val = input("Nhập số nhà bắt đầu (1-35): ").strip()
    d_val = input("Nhập số nhà kết thúc (1-35): ").strip()

    bd = tim_vi_tri(b_val, mini_map)
    dc = tim_vi_tri(d_val, mini_map)

    if not bd or not dc:
        print("❌ LỖI: Số nhà không hợp lệ!")
    else:
        path, cost, opened = a_sao(bd, dc, mini_map)
        
        if path:
            print(f"\n✅ TÌM THẤY ĐƯỜNG ĐI! Đang tạo hình ảnh báo cáo...")
            # Gọi hàm vẽ mới với đầy đủ tham số
            ve_do_thi_chuyen_nghiep(mini_map, path, bd, dc, cost, opened)
        else:
            print("❌ KHÔNG TÌM THẤY ĐƯỜNG ĐI (Bị chặn)!")