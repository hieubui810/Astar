import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
# 2. HÀM XỬ LÝ LOGIC
# ==========================================

# Tìm tọa độ (hàng, cột) của một giá trị
def tim_vi_tri(giatri, luoi):
    for r, dong in enumerate(luoi):
        for c, o in enumerate(dong):
            if o == giatri:
                return (r, c)
    return None

# Tính khoảng cách Manhattan (Heuristic)
def heuristic(nut, dich):
    return abs(nut[0] - dich[0]) + abs(nut[1] - dich[1])

# Xác định các ô lân cận hợp lệ (Xử lý vật cản)
def hang_xom(nut, batdau, dich, luoi):
    dong, cot = nut
    huong = [(-1,0), (1,0), (0,-1), (0,1)]
    ket_qua = []
    
    for dx, dy in huong:
        nd, nc = dong + dx, cot + dy
        if 0 <= nd < len(luoi) and 0 <= nc < len(luoi[0]):
            o = luoi[nd][nc]
            # Chỉ đi vào đường (.) hoặc điểm Start/End
            if o == "." or (nd, nc) == batdau or (nd, nc) == dich:
                ket_qua.append((nd, nc))
    return ket_qua

# Triển khai thuật toán A*
def a_sao(batdau, dich, luoi):
    mo = [] 
    heapq.heappush(mo, (0, batdau))
    dong = set()
    
    g_diem = {batdau: 0}
    f_diem = {batdau: heuristic(batdau, dich)}
    cha = {batdau: None}
    
    so_node_duyet = 0 

    while mo:
        _, nut_hien_tai = heapq.heappop(mo)
        
        if nut_hien_tai in dong:
            continue
            
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
            if xom in dong:
                continue
                
            g_tam = g_diem[nut_hien_tai] + 1
            if xom not in g_diem or g_tam < g_diem[xom]:
                g_diem[xom] = g_tam
                f_diem[xom] = g_tam + heuristic(xom, dich)
                cha[xom] = nut_hien_tai
                heapq.heappush(mo, (f_diem[xom], xom))
                
    return None, 0, so_node_duyet

# ==========================================
# 3. HÀM HIỂN THỊ & VẼ ĐỒ THỊ
# ==========================================

# Vẽ đồ thị đường đi và lưu ảnh (Matplotlib)
def ve_do_thi_chuyen_nghiep(luoi, duong_di, batdau, dich, show_grid_labels=True):
    rows = len(luoi)
    cols = len(luoi[0])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Tạo nền (0: Trắng, 1: Xanh nhạt)
    color_map = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if luoi[r][c] != '.' and (r, c) != batdau and (r, c) != dich:
                color_map[r][c] = 1 
    
    cmap = mcolors.ListedColormap(['white', '#6FA8DC']) 
    ax.imshow(color_map, cmap=cmap, origin='upper')

    # Kẻ lưới
    ax.set_xticks([x - 0.5 for x in range(1, cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, rows)], minor=True)
    ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Vẽ đường đi
    if duong_di:
        path_y = [p[0] for p in duong_di]
        path_x = [p[1] for p in duong_di]
        ax.plot(path_x, path_y, color='#F29F3F', linewidth=5, zorder=2, label='Đường đi')

    # (ĐÃ XÓA) Phần vẽ chữ A và B to để tránh che số

    # Hiển thị số nhà lên tất cả các ô (bao gồm start/end)    
    for r in range(rows):
        for c in range(cols):
            txt = luoi[r][c]
            if txt != '.':
                # Vẽ số màu đen, font vừa phải
                ax.text(c, r, txt, ha='center', va='center', color='black', fontsize=10, fontweight='bold', zorder=3) 

    # Hiển thị tọa độ trục
    if show_grid_labels:
        ax.set_xticks(range(cols)); ax.set_xticklabels([f"{i}" for i in range(cols)], fontsize=9)
        ax.set_yticks(range(rows)); ax.set_yticklabels([f"{i}" for i in range(rows)], fontsize=9)
    else:
        ax.set_xticks([]); ax.set_yticks([])

    ax.set_xlim(-0.5, cols-0.5); ax.set_ylim(rows-0.5, -0.5)
    plt.title(f"A* Pathfinding: {luoi[batdau[0]][batdau[1]]} -> {luoi[dich[0]][dich[1]]}", pad=20, fontsize=12)
    
    plt.savefig("ket_qua_a_star.png", dpi=300, bbox_inches='tight')
    plt.show()

# In bản đồ ra Terminal
def in_ban_do_goc(luoi):
    print("\n[BẢN ĐỒ HIỆN TẠI]")
    cols = len(luoi[0])
    print("      " + "".join(f"{c:<4}" for c in range(cols)))
    print("    " + "="*(cols*4))
    for r, dong in enumerate(luoi):
        print(f"H {r:<2} |" + "".join(f"{o:>3} " for o in dong))
    print("\n")

# ==========================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    in_ban_do_goc(mini_map) 
    
    print("--- TÌM ĐƯỜNG A* (House Delivery) ---")
    # Hiển thị rõ giới hạn input cho người dùng
    b_val = input("Nhập số nhà bắt đầu (Giới hạn: 1-35): ").strip()
    d_val = input("Nhập số nhà kết thúc (Giới hạn: 1-35): ").strip()

    bd = tim_vi_tri(b_val, mini_map)
    dc = tim_vi_tri(d_val, mini_map)

    if not bd or not dc:
        print("❌ LỖI: Số nhà không tồn tại hoặc nằm ngoài giới hạn 1-35!")
    else:
        path, cost, opened = a_sao(bd, dc, mini_map)
        
        if path:
            print(f"\n✅ TÌM THẤY ĐƯỜNG ĐI")
            print(f"🔹 Chi phí (Cost): {cost}")
            print(f"🔹 Node đã duyệt: {opened}")
            print("Đang hiển thị bản đồ...")
            ve_do_thi_chuyen_nghiep(mini_map, path, bd, dc)
        else:
            print("❌ KHÔNG TÌM THẤY ĐƯỜNG ĐI (Bị chặn)!")