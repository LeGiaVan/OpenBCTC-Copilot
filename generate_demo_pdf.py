from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

out_dir = Path('data/VNM_2025')
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / 'VNM_2025.pdf'

c = canvas.Canvas(str(out_file), pagesize=A4)
width, height = A4

c.setTitle('Báo cáo tài chính hợp nhất năm 2025')
c.setFont('Helvetica-Bold', 18)
c.drawString(60, height - 60, 'Viet Nam Dairy Products JSC')
c.setFont('Helvetica', 12)
c.drawString(60, height - 90, 'Báo cáo tài chính hợp nhất năm 2025')

lines = [
    '1. Tổng tài sản: 45.952,497 tỷ đồng',
    '2. Nợ phải trả: 18.103,550 tỷ đồng',
    '3. Vốn chủ sở hữu: 27.848,947 tỷ đồng',
    '4. ROE: 17,6%',
    '5. Chỉ số thanh toán hiện hành: 1,75 lần',
    '6. Trong thuyết minh, doanh nghiệp ghi nhận biến động của các khoản phải thu',
    '   và đầu tư dài hạn trong tương quan với dòng tiền hoạt động.',
    '7. Tài sản lưu động tăng 12,4% so với kỳ trước.',
]

y = height - 140
for line in lines:
    c.setFont('Helvetica', 11)
    c.drawString(60, y, line)
    y -= 22

c.showPage()
c.save()
print(f'Created {out_file} ({out_file.stat().st_size} bytes)')
