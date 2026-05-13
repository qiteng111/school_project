import os
import shutil
import fitz
import re
from docx import Document
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import webbrowser
from typing import List, Tuple, Optional
import unicodedata

# 全半角空格映射（用于统一处理）
SPACE_CHARS = re.compile(r'[\u0020\u3000\t\n\r\f\v]+')  # 半角空格、全角空格、制表符、换行符等

class SensitiveFileDetector:
    def __init__(self):
        self.sensitive_patterns: List[re.Pattern] = []
    
    def compile_sensitive_patterns(self, sensitive_words: List[str]) -> None:
        """编译敏感词为正则表达式（支持中间任意空字符）"""
        self.sensitive_patterns.clear()
        for word in sensitive_words:
            if not word.strip():
                continue
            # 将敏感词每个字符之间插入匹配任意空字符的表达式
            pattern_str = r''.join([f'{re.escape(c)}[\u0020\u3000\t\n\r\f\v]*' for c in word])
            # 匹配整个词（可选，根据需求调整）
            pattern = re.compile(pattern_str, re.IGNORECASE)
            self.sensitive_patterns.append(pattern)
    
    def contains_sensitive_content(self, text: str) -> bool:
        """检查文本是否包含敏感词（正则匹配）"""
        if not text or not self.sensitive_patterns:
            return False
        # 统一处理空字符（替换为半角空格，方便匹配）
        normalized_text = SPACE_CHARS.sub(' ', text)
        for pattern in self.sensitive_patterns:
            if pattern.search(normalized_text):
                return True
        return False

class FileProcessor:
    def __init__(self, detector: SensitiveFileDetector):
        self.detector = detector
        self.sensitive_files: List[Tuple[str, str, str]] = []  # (文件路径, 敏感位置, 匹配的敏感词)
        self.processed_count = 0
    
    def reset(self):
        """重置处理状态"""
        self.sensitive_files.clear()
        self.processed_count = 0
    
    def read_pdf_text(self, file_path: str) -> str:
        """读取PDF文件文本内容"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"读取PDF文件错误 {file_path}: {e}")
            return ""
    
    def read_docx_text(self, file_path: str) -> str:
        """读取DOCX文件文本内容"""
        try:
            doc = Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text
            return text
        except Exception as e:
            print(f"读取DOCX文件错误 {file_path}: {e}")
            return ""
    
    def check_path_sensitive(self, path: str) -> Tuple[bool, str]:
        """检查路径（文件夹名+文件名）是否包含敏感词"""
        # 分割路径为各个部分
        parts = []
        temp_path = path
        while True:
            dir_name, base_name = os.path.split(temp_path)
            if base_name:
                parts.append(base_name)
            if dir_name == temp_path:
                break
            temp_path = dir_name
        
        # 检查每个部分
        for part in parts:
            if self.detector.contains_sensitive_content(part):
                return True, f"路径包含敏感词（{part}）"
        return False, ""
    
    def check_file_content(self, file_path: str) -> Tuple[bool, str]:
        """检查文件内容是否包含敏感词"""
        file_ext = os.path.splitext(file_path)[1].lower()
        text = ""
        if file_ext == ".docx":
            text = self.read_docx_text(file_path)
        elif file_ext == ".pdf":
            text = self.read_pdf_text(file_path)
        else:
            return False, "不支持的文件类型"
        
        if self.detector.contains_sensitive_content(text):
            return True, "文件内容包含敏感词"
        return False, ""
    
    def process_file(self, file_path: str, mode: str, dest_dir: Optional[str]) -> None:
        """处理单个文件"""
        self.processed_count += 1
        sensitive = False
        sensitive_reason = ""
        
        # 1. 检查路径（文件夹名+文件名）
        path_sensitive, path_reason = self.check_path_sensitive(file_path)
        if path_sensitive:
            sensitive = True
            sensitive_reason = path_reason
        
        # 2. 检查文件内容（仅支持PDF和DOCX）
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in [".pdf", ".docx"]:
            content_sensitive, content_reason = self.check_file_content(file_path)
            if content_sensitive:
                sensitive = True
                if sensitive_reason:
                    sensitive_reason += " | " + content_reason
                else:
                    sensitive_reason = content_reason
        
        # 记录敏感文件
        if sensitive:
            # 找到匹配的敏感词（用于显示）
            matched_word = self._get_matched_word(file_path, path_sensitive, content_sensitive)
            self.sensitive_files.append((file_path, sensitive_reason, matched_word))
            return
        
        # 复制模式：拷贝非敏感文件
        if mode == "copy" and dest_dir:
            self._copy_non_sensitive_file(file_path, dest_dir)
    
    def _get_matched_word(self, file_path: str, path_sensitive: bool, content_sensitive: bool) -> str:
        """获取匹配的敏感词（用于报告显示）"""
        text_to_check = ""
        if path_sensitive:
            text_to_check = file_path
        elif content_sensitive:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == ".docx":
                text_to_check = self.read_docx_text(file_path)
            elif file_ext == ".pdf":
                text_to_check = self.read_pdf_text(file_path)
        
        normalized_text = SPACE_CHARS.sub(' ', text_to_check)
        for pattern in self.detector.sensitive_patterns:
            match = pattern.search(normalized_text)
            if match:
                return match.group()
        return "未知敏感词"
    
    def _copy_non_sensitive_file(self, file_path: str, dest_dir: str) -> None:
        """拷贝非敏感文件到目标目录"""
        try:
            # 保持原目录结构
            rel_path = os.path.relpath(file_path, self.source_dir)
            dest_path = os.path.join(dest_dir, rel_path)
            
            # 创建目标目录
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # 处理同名文件
            counter = 1
            original_dest_path = dest_path
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(original_dest_path)
                dest_path = f"{name}_{counter}{ext}"
                counter += 1
            
            shutil.copy2(file_path, dest_path)
        except Exception as e:
            print(f"拷贝文件错误 {file_path}: {e}")
    
    def process_directory(self, source_dir: str, mode: str, dest_dir: Optional[str]) -> None:
        """处理整个目录"""
        self.source_dir = source_dir
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self.process_file(file_path, mode, dest_dir)

class SensitiveFileDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("文件脱敏检测工具 v1.0")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # 初始化核心组件
        self.detector = SensitiveFileDetector()
        self.processor = FileProcessor(self.detector)
        
        # 变量定义
        self.source_dir_var = tk.StringVar()
        self.dest_dir_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="check")  # check: 检查模式, copy: 复制模式
        self.sensitive_words_var = tk.Text()
        
        # 构建UI
        self._build_ui()
    
    def _build_ui(self):
        """构建用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 1. 路径选择区域
        path_frame = ttk.LabelFrame(main_frame, text="路径设置", padding="10")
        path_frame.pack(fill=tk.X, pady=5)
        
        # 源路径
        ttk.Label(path_frame, text="源文件夹:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(path_frame, textvariable=self.source_dir_var, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(path_frame, text="浏览", command=self._select_source_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # 目标路径（仅复制模式显示）
        self.dest_label = ttk.Label(path_frame, text="目标文件夹:")
        self.dest_entry = ttk.Entry(path_frame, textvariable=self.dest_dir_var, width=60)
        self.dest_button = ttk.Button(path_frame, text="浏览", command=self._select_dest_dir)
        
        self.dest_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.dest_entry.grid(row=1, column=1, padx=5, pady=5)
        self.dest_button.grid(row=1, column=2, padx=5, pady=5)
        
        # 2. 模式选择区域
        mode_frame = ttk.LabelFrame(main_frame, text="运行模式", padding="10")
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(mode_frame, text="检查模式（仅检测不复制）", variable=self.mode_var, value="check", command=self._toggle_mode).grid(row=0, column=0, padx=10, pady=5)
        ttk.Radiobutton(mode_frame, text="复制模式（检测并复制非敏感文件）", variable=self.mode_var, value="copy", command=self._toggle_mode).grid(row=0, column=1, padx=10, pady=5)
        
        # 3. 敏感词设置区域
        sensitive_frame = ttk.LabelFrame(main_frame, text="敏感词设置（每行一个，支持正则匹配）", padding="10")
        sensitive_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(sensitive_frame, text="敏感词列表（例如：机密、秘密、内部）:").pack(anchor=tk.W, padx=5, pady=2)
        self.sensitive_words_var = scrolledtext.ScrolledText(sensitive_frame, height=8, width=80)
        self.sensitive_words_var.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # 默认敏感词
        default_words = "机密\n秘密\n绝密\n内部"
        self.sensitive_words_var.insert(tk.END, default_words)
        
        # 4. 控制按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="开始处理", command=self._start_process)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="清空结果", command=self._clear_results)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # 5. 进度显示区域
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(main_frame, text="就绪")
        self.status_label.pack(anchor=tk.W, padx=5)
        
        # 6. 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="检测结果（敏感文件列表）", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 结果表格
        columns = ("file_path", "reason", "matched_word")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("file_path", text="文件路径", command=lambda: self._sort_treeview(0, False))
        self.tree.heading("reason", text="敏感原因", command=lambda: self._sort_treeview(1, False))
        self.tree.heading("matched_word", text="匹配的敏感词", command=lambda: self._sort_treeview(2, False))
        
        # 设置列宽
        self.tree.column("file_path", width=500, anchor=tk.W)
        self.tree.column("reason", width=250, anchor=tk.W)
        self.tree.column("matched_word", width=150, anchor=tk.W)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定双击打开文件
        self.tree.bind("<Double-1>", self._open_selected_file)
        
        # 初始化模式
        self._toggle_mode()
    
    def _toggle_mode(self):
        """切换运行模式（显示/隐藏目标路径）"""
        if self.mode_var.get() == "copy":
            self.dest_label.config(state=tk.NORMAL)
            self.dest_entry.config(state=tk.NORMAL)
            self.dest_button.config(state=tk.NORMAL)
        else:
            self.dest_label.config(state=tk.DISABLED)
            self.dest_entry.config(state=tk.DISABLED)
            self.dest_button.config(state=tk.DISABLED)
    
    def _select_source_dir(self):
        """选择源文件夹"""
        dir_path = filedialog.askdirectory(title="选择源文件夹")
        if dir_path:
            self.source_dir_var.set(dir_path)
    
    def _select_dest_dir(self):
        """选择目标文件夹"""
        dir_path = filedialog.askdirectory(title="选择目标文件夹")
        if dir_path:
            self.dest_dir_var.set(dir_path)
    
    def _start_process(self):
        """开始处理文件"""
        # 验证输入
        source_dir = self.source_dir_var.get().strip()
        if not source_dir or not os.path.isdir(source_dir):
            messagebox.showerror("错误", "请选择有效的源文件夹！")
            return
        
        mode = self.mode_var.get()
        dest_dir = None
        if mode == "copy":
            dest_dir = self.dest_dir_var.get().strip()
            if not dest_dir:
                messagebox.showerror("错误", "复制模式请选择目标文件夹！")
                return
            # 创建目标文件夹
            os.makedirs(dest_dir, exist_ok=True)
        
        # 获取敏感词
        sensitive_words = self.sensitive_words_var.get("1.0", tk.END).strip().split("\n")
        sensitive_words = [word.strip() for word in sensitive_words if word.strip()]
        if not sensitive_words:
            messagebox.showerror("错误", "请输入至少一个敏感词！")
            return
        
        # 编译敏感词正则表达式
        self.detector.compile_sensitive_patterns(sensitive_words)
        
        # 重置状态
        self.processor.reset()
        self._clear_results()
        self.start_button.config(state=tk.DISABLED)
        self.status_label.config(text="正在处理...")
        self.root.update_idletasks()
        
        try:
            # 先计算总文件数（用于进度条）
            total_files = self._count_files(source_dir)
            if total_files == 0:
                messagebox.showinfo("提示", "源文件夹中没有可处理的文件（仅支持PDF和DOCX）！")
                return
            
            # 处理文件
            self.processor.process_directory(source_dir, mode, dest_dir)
            
            # 更新进度
            self.progress_var.set(100)
            self.status_label.config(text=f"处理完成！共处理 {self.processor.processed_count} 个文件，发现 {len(self.processor.sensitive_files)} 个敏感文件")
            
            # 显示结果
            self._show_results()
            
            # 复制模式提示
            if mode == "copy":
                messagebox.showinfo("提示", f"复制完成！非敏感文件已保存到：{dest_dir}")
        
        except Exception as e:
            messagebox.showerror("错误", f"处理过程中发生错误：{str(e)}")
            self.status_label.config(text=f"处理失败：{str(e)}")
        finally:
            self.start_button.config(state=tk.NORMAL)
    
    def _count_files(self, directory: str) -> int:
        """计算目录下可处理的文件数（PDF和DOCX）"""
        count = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in [".pdf", ".docx"]:
                    count += 1
        return count
    
    def _show_results(self):
        """显示检测结果"""
        for file_path, reason, matched_word in self.processor.sensitive_files:
            self.tree.insert("", tk.END, values=(file_path, reason, matched_word))
    
    def _clear_results(self):
        """清空结果"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.progress_var.set(0)
        self.status_label.config(text="就绪")
    
    def _open_selected_file(self, event):
        """双击打开选中的文件"""
        selected_item = self.tree.selection()
        if not selected_item:
            return
        file_path = self.tree.item(selected_item)["values"][0]
        if os.path.exists(file_path):
            webbrowser.open(f"file://{os.path.abspath(file_path)}")
        else:
            messagebox.showwarning("警告", "文件不存在或已被移动！")
    
    def _sort_treeview(self, col, reverse):
        """排序表格"""
        data = [(self.tree.set(child, col), child) for child in self.tree.get_children('')]
        data.sort(reverse=reverse)
        for index, (val, child) in enumerate(data):
            self.tree.move(child, '', index)
        self.tree.heading(col, command=lambda: self._sort_treeview(col, not reverse))

def main():
    root = tk.Tk()
    app = SensitiveFileDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()