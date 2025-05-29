import os
from PIL import Image, ImageOps
import numpy as np
import json
import re
from paddleocr import PaddleOCR
from datetime import datetime

class ReceiptProcessor:
    def __init__(self, lang='en', use_gpu=False):
        """
        Initialize the receipt processor with PaddleOCR
        
        Args:
            lang (str): Language for OCR, use 'en' for English or 'ch' for Chinese
            use_gpu (bool): Whether to use GPU for inference
        """
        # Initialize PaddleOCR with specified language and device
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)
        
        # Define regex patterns for data extraction
        self.date_pattern = r'(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{2,4})'
        self.time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?)'
        
        # Updated price pattern for Indonesian Rupiah format
        self.price_pattern = r'(?:Rp\.?|Rp)?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{0,3}))'
        
        # Pattern to match total-related keywords
        self.total_pattern = r'(?:total|jumlah|amount|sub\s*total|subtotal).*?(?:Rp\.?|Rp)?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{0,3}))'
        
        # Keywords to identify non-item text
        self.non_item_keywords = [
            'total', 'subtotal', 'sub total', 'pajak', 'tax', 'service', 
            'charge', 'amount', 'jumlah', 'tunai', 'cash', 'card', 'kartu',
            'debit', 'credit', 'kredit', 'change', 'kembalian', 'kembali',
            'date', 'time', 'tanggal', 'waktu', 'receipt', 'invoice',
            'customer', 'pelanggan', 'thank', 'terima kasih', 'sales',
            'queue', 'antrian', 'no.', 'nomor', 'table', 'meja', 'rcpt',
            'bill', 'order', 'cashier', 'kasir', 'pax', 'guest', 'phone',
            'telp', 'address', 'alamat', 'jalan', 'street', 'kota', 'city',
            'provinsi', 'province', 'kode pos', 'postal code', 'zip',
            'npwp', 'tax id', 'produk', 'product', 'item', 'qty', 'quantity',
            'harga', 'price', 'disc', 'discount', 'potongan', 'ppn', 'vat',
            'dpp', 'pbj', 'gedung', 'building', 'menara', 'tower', 'plaza',
            'mall', 'kec', 'kecamatan', 'district', 'kelurahan', 'desa',
            'village', 'rt', 'rw', 'blok', 'block', 'jl', 'jln', 'jalan',
            'street', 'avenue', 'lane', 'road', 'boulevard', 'highway',
            'kabupaten', 'regency', 'kota', 'city', 'provinsi', 'province',
            'indonesia', 'jakarta', 'bandung', 'surabaya', 'medan', 'makassar',
            'semarang', 'palembang', 'tangerang', 'depok', 'bekasi', 'batam',
            'pekanbaru', 'bogor', 'padang', 'malang', 'samarinda', 'tasikmalaya',
            'pontianak', 'banjarmasin', 'balikpapan', 'manado', 'denpasar',
            'serang', 'jambi', 'bengkulu', 'ambon', 'palu', 'mataram',
            'kupang', 'jayapura', 'ternate', 'tanjung pinang', 'pangkal pinang',
            'gorontalo', 'mamuju', 'kendari', 'banda aceh', 'tanjung selor',
            'manokwari', 'sofifi', 'mamuju', 'nabire', 'merauke', 'sorong',
            'biak', 'timika', 'wamena', 'fakfak', 'serui', 'tembagapura',
            'subtota1', 'subtota2', 'subtota3', 'subtota4', 'subtota5',
            'subtota6', 'subtota7', 'subtota8', 'subtota9', 'subtota0'
        ]
        
        # Define categories for item classification
        self.categories = {
            'FOOD': ['nasi', 'mie', 'ayam', 'daging', 'ikan', 'sayur', 'sushi', 'ramen', 'butadon', 'gyoza', 'chicken', 'beef', 'fish', 'rice', 'soup'],
            'BEVERAGE': ['teh', 'tea', 'kopi', 'coffee', 'air', 'water', 'jus', 'juice', 'milk', 'susu', 'cappuccino', 'latte', 'espresso', 'boba', 'bubble'],
            'SNACK': ['keripik', 'chips', 'kue', 'cake', 'coklat', 'chocolate', 'permen', 'candy', 'cookie', 'biskuit', 'biscuit'],
            'GROCERY': ['beras', 'rice', 'minyak', 'oil', 'gula', 'sugar', 'tepung', 'flour', 'telur', 'egg'],
            'HOUSEHOLD': ['sabun', 'soap', 'deterjen', 'detergent', 'tissue', 'pembersih', 'cleaner', 'sikat', 'brush'],
            'PERSONAL_CARE': ['shampo', 'shampoo', 'sikat', 'brush', 'pasta', 'gigi', 'tooth', 'deodorant', 'lotion']
        }
        
    def preprocess_image(self, image_path):
        """
        Preprocess image using PIL instead of OpenCV to avoid libGL issues
        """
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = ImageOps.autocontrast(image)  # Enhance contrast
        image = image.resize((int(image.width * 1.5), int(image.height * 1.5)))  # Resize to make text more legible

        # Save to temporary path
        preprocessed_path = f"{os.path.splitext(image_path)[0]}_preprocessed.jpg"
        image.save(preprocessed_path)
        print(f"Preprocessed image saved to {preprocessed_path}")

        return preprocessed_path
    
    def extract_text(self, image_path, preprocess=True):
        """
        Extract text from receipt image using PaddleOCR
        
        Args:
            image_path (str): Path to the receipt image
            preprocess (bool): Whether to preprocess the image
            
        Returns:
            list: List of detected text and their positions
            str: Full text from the receipt
        """
        # Preprocess the image if required
        if preprocess:
            preprocessed_path = self.preprocess_image(image_path)
            result = self.ocr.ocr(preprocessed_path, cls=True)
        else:
            # Run OCR directly on original image
            result = self.ocr.ocr(image_path, cls=True)
        
        if not result or not result[0]:
            return [], ""
        
        # Extract text and positions
        text_results = []
        full_text = []
        
        for line in result[0]:
            position = line[0]
            text = line[1][0]
            confidence = line[1][1]
            
            text_results.append({
                'text': text,
                'position': position,
                'confidence': float(confidence)
            })
            
            full_text.append(text)
        
        return text_results, '\n'.join(full_text)
    
    def identify_receipt_type(self, text):
        """
        Identify the type of receipt based on text content
        
        Args:
            text (str): Full text from the receipt
            
        Returns:
            str: Type of receipt
        """
        text_lower = text.lower()
        
        # Check for known restaurant/store patterns
        if 'gomachi' in text_lower or 'japanese ramen' in text_lower or 'gemachi' in text_lower:
            return 'GOMACHI'
        elif 'chatime' in text_lower or 'milk tea' in text_lower or 'chatine' in text_lower:
            return 'CHATIME'
        elif 'sushigo' in text_lower or 'one price sushi' in text_lower:
            return 'SUSHIGO'
        elif 'hokben' in text_lower or 'hoka ichiman' in text_lower:
            return 'HOKBEN'
        elif 'indomaret' in text_lower or 'indomarco' in text_lower:
            return 'INDOMARET'
        elif 'warung ce' in text_lower or 'goldfinch' in text_lower:
            return 'WARUNG_CE'
        
        return 'UNKNOWN'
    
    def extract_merchant_name(self, text_results, receipt_type):
        """
        Extract merchant name from the receipt
        
        Args:
            text_results (list): List of detected text and their positions
            receipt_type (str): Type of receipt
            
        Returns:
            str: Merchant name
        """
        # Sort text by y-coordinate to get top lines
        sorted_text = sorted(text_results, key=lambda x: x['position'][0][1])
        
        # Check first few lines for merchant name
        for i in range(min(5, len(sorted_text))):
            text = sorted_text[i]['text']
            
            # Check for known merchant names based on receipt type
            if receipt_type == 'GOMACHI' and ('Gomachi' in text or 'GOMACHI' in text or 'Gemachi' in text):
                return "Gomachi"
            elif receipt_type == 'CHATIME' and ('Chatime' in text or 'CHATIME' in text or 'Chatine' in text):
                return "Chatime"
            elif receipt_type == 'SUSHIGO' and ('SUSHIGO' in text or 'Sushigo' in text):
                return text
            elif receipt_type == 'HOKBEN' and ('HOKBEN' in text or 'HokBen' in text):
                return text
            elif receipt_type == 'INDOMARET' and ('INDOMARET' in text or 'Indomaret' in text):
                return text
            elif receipt_type == 'WARUNG_CE' and ('Warung Ce' in text or 'WARUNG CE' in text):
                return text
        
        # If no specific merchant name found, return first line
        return sorted_text[0]['text'] if sorted_text else "Unknown Merchant"
    
    def standardize_date_format(self, date_str):
        """
        Standardize date format to DD/MM/YYYY
        
        Args:
            date_str (str): Date string in various formats
            
        Returns:
            str: Standardized date string
        """
        if not date_str:
            return ""
        
        # Try different date formats
        date_formats = [
            '%d/%m/%Y', '%d/%m/%y', '%d-%m-%Y', '%d-%m-%y', 
            '%d.%m.%Y', '%d.%m.%y', '%d %B %Y', '%d %b %Y',
            '%Y/%m/%d', '%y/%m/%d', '%Y-%m-%d', '%y-%m-%d',
            '%Y.%m.%d', '%y.%m.%d', '%B %d %Y', '%b %d %Y'
        ]
        
        # Try to parse the date
        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                # Return in standard format DD/MM/YYYY
                return date_obj.strftime('%d/%m/%Y')
            except ValueError:
                continue
        
        # If parsing fails, return the original string
        return date_str
    
    def standardize_time_format(self, time_str):
        """
        Standardize time format to HH:MM
        
        Args:
            time_str (str): Time string in various formats
            
        Returns:
            str: Standardized time string
        """
        if not time_str:
            return ""
        
        # Try different time formats
        time_formats = ['%H:%M:%S', '%H:%M', '%I:%M:%S %p', '%I:%M %p']
        
        # Try to parse the time
        for fmt in time_formats:
            try:
                time_obj = datetime.strptime(time_str, fmt)
                # Return in standard format HH:MM
                return time_obj.strftime('%H:%M')
            except ValueError:
                continue
        
        # If parsing fails, return the original string
        return time_str
    
    def extract_date_time(self, full_text):
        """
        Extract date and time from receipt text
        
        Args:
            full_text (str): Full text from the receipt
            
        Returns:
            tuple: (date, time)
        """
        date = ""
        time = ""
        
        # Search for date pattern
        date_match = re.search(self.date_pattern, full_text)
        if date_match:
            date = date_match.group(1)
            date = self.standardize_date_format(date)
        
        # Search for time pattern
        time_match = re.search(self.time_pattern, full_text)
        if time_match:
            time = time_match.group(1)
            time = self.standardize_time_format(time)
        
        return date, time
    
    def is_valid_item_name(self, text):
        """
        Check if a text is a valid item name using a blacklist approach
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if the text is a valid item name, False otherwise
        """
        # Check if text is too short
        if len(text) < 2:
            return False
        
        # Check if text is mostly numbers or contains only numbers
        if re.match(r'^\d+(\s*\d+)*$', text):
            return False
        
        # Check if text contains price format
        if re.search(r'Rp\.?\s*\d+', text):
            return False
        
        # Check for receipt-specific keywords that are definitely not items
        non_item_patterns = [
            r'^rcpt', r'^receipt', r'^invoice', r'^bill', r'^order',
            r'^subtotal', r'^sub\s*total', r'^total', r'^pajak', r'^tax',
            r'^service', r'^charge', r'^amount', r'^jumlah', r'^tunai',
            r'^cash', r'^card', r'^kartu', r'^debit', r'^credit', r'^kredit',
            r'^change', r'^kembalian', r'^kembali', r'^date', r'^time',
            r'^tanggal', r'^waktu', r'^customer', r'^pelanggan', r'^thank',
            r'^terima\s*kasih', r'^sales', r'^queue', r'^antrian', r'^no\.',
            r'^nomor', r'^table', r'^meja', r'^produk', r'^product',
            r'^qty', r'^quantity', r'^harga', r'^price', r'^disc', r'^discount',
            r'^potongan', r'^ppn', r'^vat', r'^dpp', r'^pbj', r'^subtota\d',
            r'^kec\.', r'^kel\.', r'^jl\.', r'^jln\.', r'^rt\.', r'^rw\.',
            r'^blok', r'^gedung', r'^menara', r'^plaza', r'^mall', r'^edc', r'^bca', r'^scrp', r'^pb1rp' , r'^sub'
        ]
        
        text_lower = text.lower()
        for pattern in non_item_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check if text contains any of the non-item keywords
        for keyword in self.non_item_keywords:
            if keyword in text_lower:
                return False
        
        # Check if text contains date pattern
        if re.search(r'\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}', text):
            return False
        
        # Check if text contains time pattern
        if re.search(r'\d{1,2}:\d{2}(?::\d{2})?', text):
            return False
        
        # Check if text contains location-specific patterns (postal codes, etc.)
        if re.search(r'\d{5}', text):  # 5-digit postal code
            return False
        
        # Check if the first word is "less" (likely a modifier, not an item)
        words = text.strip().split()
        if words and words[0].lower() == "less":
            return False
        if len(words) == 2 and words[1].lower() == "less":
            return False
        
        # If it passed all filters, consider it a valid item
        return True
    
    def fix_price_format(self, price_str):
        """
        Fix the price format for Indonesian Rupiah
        
        Args:
            price_str (str): Price string extracted from receipt
            
        Returns:
            float: Corrected price value
        """
        # Remove non-numeric characters except for decimal point and comma
        price_str = re.sub(r'[^\d.,]', '', price_str)
        
        # Handle different decimal separators
        if ',' in price_str and '.' in price_str:
            # If both comma and period are present, assume period is thousands separator
            # and comma is decimal separator (e.g., 1.000,00)
            price_str = price_str.replace('.', '')
            price_str = price_str.replace(',', '.')
        elif ',' in price_str:
            # If only comma is present, assume it's a decimal separator or thousands separator
            # based on position
            if price_str.endswith(',00') or price_str.endswith(',0'):
                # Likely a decimal separator
                price_str = price_str.replace(',', '.')
            else:
                # Likely a thousands separator
                price_str = price_str.replace(',', '')
        
        # Convert to float
        try:
            price = float(price_str)
        except ValueError:
            print(f"Warning: Could not convert price '{price_str}' to float")
            return 0.0
        
        # Check if the price seems too small for Indonesian Rupiah
        # Most items in Indonesia cost at least thousands of Rupiah
        if 0 < price < 1000:
            # Multiply by 1000 to correct the value
            price *= 1000
        
        return price
    
    def extract_quantity_from_name(self, item_name):
        """
        Extract quantity from item name if it starts with a number
        
        Args:
            item_name (str): Item name that might start with quantity
            
        Returns:
            tuple: (quantity, cleaned_item_name)
        """
        # Check for patterns like "2GYOZA" or "4 PIRING"
        quantity_match = re.match(r'^(\d+)(\s*)(.+)$', item_name)
        
        if quantity_match:
            quantity = int(quantity_match.group(1))
            # Get the rest of the name without the quantity
            cleaned_name = quantity_match.group(3)
            return quantity, cleaned_name
        
        return 1, item_name
    
    def extract_items_from_chatime_receipt(self, text_results, full_text):
        """
        Special handler for Chatime receipts which have a specific format
        
        Args:
            text_results (list): List of detected text and their positions
            full_text (str): Full text from the receipt
            
        Returns:
            list: List of items with their details
        """
        items = []
        lines = full_text.split('\n')
        
        # Look for lines with the pattern "1 x ITEM_NAME (L)" followed by price
        item_pattern = r'(\d+)\s*[xX]\s*([A-Za-z\s]+)(?:$$[A-Za-z]$$)?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{0,3}))'
        
        for i, line in enumerate(lines):
            # Try to match the item pattern
            match = re.search(item_pattern, line)
            if match:
                quantity = int(match.group(1))
                item_name = match.group(2).strip()
                price_str = match.group(3)
                price = self.fix_price_format(price_str)
                
                # Check if the next line is a modifier (like "SUGAR LESS")
                if i + 1 < len(lines) and "SUGAR" in lines[i + 1]:
                    modifier = lines[i + 1].strip()
                    # Append modifier to item name
                    item_name = f"{item_name} ({modifier})"
                
                category = self.categorize_item(item_name)
                items.append({
                    'name': item_name,
                    'quantity': quantity,
                    'price': price,
                    'category': category
                })
        
        # If no items found with the pattern, try another approach
        if not items:
            # Look for lines with "x" followed by item name
            for i, line in enumerate(lines):
                if "x" in line.lower() and not any(keyword in line.lower() for keyword in ['subtotal', 'total', 'tax']):
                    # Try to extract price from this line or next line
                    price = 0
                    price_match = re.search(self.price_pattern, line)
                    if price_match:
                        price_str = price_match.group(1)
                        price = self.fix_price_format(price_str)
                    elif i + 1 < len(lines):
                        # Check next line for price
                        next_line = lines[i + 1]
                        if re.match(r'^\s*\d+(?:[.,]\d+)*\s*$', next_line.strip()):
                            price = self.fix_price_format(next_line.strip())
                    
                    if price > 0:
                        # Extract item name and quantity
                        parts = line.split('x', 1)
                        if len(parts) == 2:
                            try:
                                quantity = int(parts[0].strip())
                                item_name = parts[1].strip()
                                
                                # Remove price from item name if present
                                if price_match:
                                    item_name = line[:line.find(price_match.group(0))].strip()
                                    if 'x' in item_name:
                                        item_name = item_name.split('x', 1)[1].strip()
                                
                                category = self.categorize_item(item_name)
                                items.append({
                                    'name': item_name,
                                    'quantity': quantity,
                                    'price': price,
                                    'category': category
                                })
                            except ValueError:
                                continue
        
        return items
    
    def extract_items(self, text_results, receipt_type, full_text):
        """
        Extract items and their prices from the receipt using position context and blacklist filtering
        
        Args:
            text_results (list): List of detected text and their positions
            receipt_type (str): Type of receipt
            full_text (str): Full text from the receipt
            
        Returns:
            list: List of items with their details
        """
        # Special handling for Chatime receipts
        if receipt_type == 'CHATIME':
            return self.extract_items_from_chatime_receipt(text_results, full_text)
        items = []
        
        # Sort text by y-coordinate
        sorted_text = sorted(text_results, key=lambda x: x['position'][0][1])
        
        # Find the item section boundaries
        header_end_idx = 0
        footer_start_idx = len(sorted_text)
        
        # Find header end (look for patterns that typically appear at the end of header)
        for i, item in enumerate(sorted_text):
            text_lower = item['text'].lower()
            if any(keyword in text_lower for keyword in ['item', 'description', 'qty', 'price', 'amount', 'jumlah']):
                header_end_idx = i + 1
                break
        
        # If no header markers found, estimate header (usually first 15-20% of receipt)
        if header_end_idx == 0:
            header_end_idx = max(5, int(len(sorted_text) * 0.15))
        
        # Find footer start (look for patterns that typically appear at the start of footer)
        for i, item in enumerate(sorted_text):
            text_lower = item['text'].lower()
            if any(keyword in text_lower for keyword in ['subtotal', 'sub total', 'total', 'pajak', 'tax']):
                footer_start_idx = i
                break
        
        # If no footer markers found, estimate footer (usually last 15-20% of receipt)
        if footer_start_idx == len(sorted_text):
            footer_start_idx = min(len(sorted_text) - 5, int(len(sorted_text) * 0.85))
        
        # Process each line in the item section
        for i in range(header_end_idx, footer_start_idx):
            line = sorted_text[i]['text']
            
            # Try to extract price
            price_match = re.search(self.price_pattern, line)
            if price_match:
                price_str = price_match.group(1)
                price = self.fix_price_format(price_str)
                
                # Extract item name (text before the price)
                item_name = line[:line.find(price_match.group(0))].strip()
                
                # If item name is empty, try to get it from previous line
                if not item_name and i > 0:
                    prev_line = sorted_text[i-1]['text']
                    # Check if previous line doesn't contain price and is not a header
                    if not re.search(self.price_pattern, prev_line) and not any(keyword in prev_line.lower() for keyword in ['item', 'description', 'qty', 'price']):
                        item_name = prev_line
                
                # Extract quantity if available
                quantity = 1
                qty_match = re.search(r'(\d+)\s*[xX]', line)
                if qty_match:
                    quantity = int(qty_match.group(1))
                
                # Basic validation of item name
                if item_name and len(item_name) >= 2:
                    # Apply blacklist filtering - exclude obvious non-items
                    if self.is_valid_item_name(item_name):
                        # Extract quantity from item name if present
                        name_quantity, cleaned_item_name = self.extract_quantity_from_name(item_name)
                        
                        # If quantity was found in the name, use it instead
                        if name_quantity > 1:
                            quantity = name_quantity
                            item_name = cleaned_item_name
                            
                        # Clean item name - remove leading numbers and 'x' patterns
                        # Check if item name starts with a digit
                        item_name = re.sub(r'^\d+\s*', '', item_name)
                        
                        # Check if item name has a pattern like "2x" or "2 x" at the beginning
                        item_name = re.sub(r'\s*[xX]\s*', '', item_name)
                        
                        category = self.categorize_item(item_name)
                        items.append({
                            'name': item_name,
                            'quantity': quantity,
                            'price': price,
                            'category': category
                        })
        
        # If no items were found using the above method, try a different approach
        if not items:
            items = self.extract_items_alternative(full_text, receipt_type)
        
        return items
    
    def extract_items_alternative(self, full_text, receipt_type):
        """
        Alternative method to extract items when the primary method fails
        
        Args:
            full_text (str): Full text from the receipt
            receipt_type (str): Type of receipt
            
        Returns:
            list: List of items with their details
        """
        items = []
        lines = full_text.split('\n')
        
        # Find potential item lines
        for i, line in enumerate(lines):
            # Skip first few lines (likely header)
            if i < 3:
                continue
                
            # Skip lines that are likely not items
            if any(keyword in line.lower() for keyword in ['subtotal', 'sub total', 'total', 'pajak', 'tax']):
                continue
                
            # Try to extract price
            price_match = re.search(self.price_pattern, line)
            if price_match:
                price_str = price_match.group(1)
                price = self.fix_price_format(price_str)
                
                # Extract item name (text before the price)
                item_name = line[:line.find(price_match.group(0))].strip()
                
                # If item name is empty, try to get it from previous line
                if not item_name and i > 0:
                    prev_line = lines[i-1]
                    if self.is_valid_item_name(prev_line):
                        item_name = prev_line
                
                # Extract quantity if available
                quantity = 1
                qty_match = re.search(r'(\d+)\s*[xX]', line)
                if qty_match:
                    quantity = int(qty_match.group(1))
                
                # Check if this is a valid item
                if self.is_valid_item_name(item_name):
                    # Extract quantity from item name if present
                    name_quantity, cleaned_item_name = self.extract_quantity_from_name(item_name)
                    
                    # If quantity was found in the name, use it instead
                    if name_quantity > 1:
                        quantity = name_quantity
                        item_name = cleaned_item_name
                        
                    # Clean item name - remove leading numbers and 'x' patterns
                    # Check if item name starts with a digit
                    item_name = re.sub(r'^\d+\s*', '', item_name)
                    
                    # Check if item name has a pattern like "2x" or "2 x" at the beginning
                    item_name = re.sub(r'^\d+\s*[xX]\s*', '', item_name)
                    
                    category = self.categorize_item(item_name)
                    items.append({
                        'name': item_name,
                        'quantity': quantity,
                        'price': price,
                        'category': category
                    })
        
        # Special handling for specific receipt types
        if receipt_type == 'SUSHIGO' and not items:
            # SushiGo receipts often have a specific format
            for i, line in enumerate(lines):
                if 'PIRING' in line or 'PLATE' in line:
                    price_match = re.search(self.price_pattern, line)
                    if price_match:
                        price_str = price_match.group(1)
                        price = self.fix_price_format(price_str)
                
                        # Extract quantity if available
                        quantity = 1
                        qty_match = re.search(r'(\d+)\s*[xX]', line)
                        if qty_match:
                            quantity = int(qty_match.group(1))
                        else:
                            # Check for quantity at the beginning of the line
                            name_quantity, cleaned_line = self.extract_quantity_from_name(line.strip())
                            if name_quantity > 1:
                                quantity = name_quantity
                                line = cleaned_line
                
                        # Make sure it's a valid item
                        if self.is_valid_item_name(line.strip()):
                            # Clean the item name
                            cleaned_name = line.strip()
                            
                            # Clean item name - remove leading numbers and 'x' patterns
                            cleaned_name = re.sub(r'^\d+\s*', '', cleaned_name)
                            cleaned_name = re.sub(r'^\d+\s*[xX]\s*', '', cleaned_name)
                            
                            items.append({
                                'name': line.strip(),
                                'quantity': quantity,
                                'price': price,
                                'category': 'FOOD'
                            })
        
        elif receipt_type == 'GOMACHI' and not items:
            # Gomachi receipts often have items with specific patterns
            for i, line in enumerate(lines):
                if 'BUTADON' in line or 'RAMEN' in line or 'GYOZA' in line:
                    # Look for price in the same line or next line
                    price = 0
                    price_match = re.search(self.price_pattern, line)
                    if price_match:
                        price_str = price_match.group(1)
                        price = self.fix_price_format(price_str)
                    elif i+1 < len(lines):
                        price_match = re.search(self.price_pattern, lines[i+1])
                        if price_match:
                            price_str = price_match.group(1)
                            price = self.fix_price_format(price_str)
                    
                    if price > 0 and self.is_valid_item_name(line.strip()):
                        # Extract quantity if available
                        quantity = 1
                        qty_match = re.search(r'(\d+)\s*[xX]', line)
                        if qty_match:
                            quantity = int(qty_match.group(1))
                        else:
                            # Check for quantity at the beginning of the line
                            name_quantity, cleaned_line = self.extract_quantity_from_name(line.strip())
                            if name_quantity > 1:
                                quantity = name_quantity
                                line = cleaned_line
                                
                        # Clean the item name
                        cleaned_name = line.strip()
                        
                        # Clean item name - remove leading numbers and 'x' patterns
                        cleaned_name = re.sub(r'^\d+\s*', '', cleaned_name)
                        cleaned_name = re.sub(r'^\d+\s*[xX]\s*', '', cleaned_name)
                
                        items.append({
                            'name': line.strip(),
                            'quantity': quantity,
                            'price': price,
                            'category': 'FOOD'
                        })
        
        return items
    
    def categorize_item(self, item_name):
        """
        Categorize an item based on its name
        
        Args:
            item_name (str): Name of the item
            
        Returns:
            str: Category of the item
        """
        item_lower = item_name.lower()
        
        for category, keywords in self.categories.items():
            if any(keyword in item_lower for keyword in keywords):
                return category
        
        return 'OTHER'
    
    def extract_totals(self, full_text):
        """
        Extract subtotal, tax, service charge, and total from receipt text
        
        Args:
            full_text (str): Full text from the receipt
            
        Returns:
            dict: Dictionary with subtotal, tax, service_charge, and total
        """
        result = {
            'subtotal': None,
            'tax': None,
            'service_charge': None,
            'total': None
        }
        
        # Extract subtotal
        subtotal_match = re.search(r'(?:subtotal|sub\s*total).*?(?:Rp\.?|Rp)?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{0,3}))', full_text, re.IGNORECASE)
        if subtotal_match:
            subtotal_str = subtotal_match.group(1)
            result['subtotal'] = self.fix_price_format(subtotal_str)
        
        # Extract tax
        tax_match = re.search(r'(?:tax|pajak|pb1).*?(?:Rp\.?|Rp)?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{0,3}))', full_text, re.IGNORECASE)
        if tax_match:
            tax_str = tax_match.group(1)
            result['tax'] = self.fix_price_format(tax_str)
        
        # Extract service charge
        service_match = re.search(r'(?:service|charge).*?(?:Rp\.?|Rp)?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{0,3}))', full_text, re.IGNORECASE)
        if service_match:
            service_str = service_match.group(1)
            result['service_charge'] = self.fix_price_format(service_str)
        
        # Extract total
        total_match = re.search(r'(?:^|\s)total.*?(?:Rp\.?|Rp)?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{0,3}))', full_text, re.IGNORECASE)
        if total_match:
            total_str = total_match.group(1)
            result['total'] = self.fix_price_format(total_str)
        
        return result
    
    def calculate_total(self, items, totals):
        """
        Calculate or verify total from items if not found in receipt
        
        Args:
            items (list): List of items with their details
            totals (dict): Dictionary with extracted totals
            
        Returns:
            float: Total price
        """
        # If total is already extracted, return it
        if totals['total'] is not None:
            return totals['total']
        
        # Calculate total from items
        items_total = sum(item['price'] for item in items)
        
        # Add tax and service charge if available
        if totals['tax'] is not None:
            items_total += totals['tax']
        
        if totals['service_charge'] is not None:
            items_total += totals['service_charge']
        
        return items_total
    
    def extract_payment_method(self, full_text):
        """
        Extract payment method from receipt text
        
        Args:
            full_text (str): Full text from the receipt
            
        Returns:
            str: Payment method
        """
        text_lower = full_text.lower()
        
        payment_methods = {
            'CASH': ['cash', 'tunai', 'uang tunai'],
            'CARD': ['card', 'kartu', 'edc'],
            'DEBIT': ['debit'],
            'CREDIT': ['credit', 'kredit', 'cc'],
            'QR_PAYMENT': ['qr', 'qris'],
            'OVO': ['ovo'],
            'GOPAY': ['gopay', 'gojek'],
            'DANA': ['dana'],
            'BCA': ['bca'],
            'MANDIRI': ['mandiri']
        }
        
        for method, keywords in payment_methods.items():
            if any(keyword in text_lower for keyword in keywords):
                return method
        
        return 'UNKNOWN'
    
    def process_receipt(self, image_path, preprocess=True):
        """
        Process a receipt image and extract all relevant information
        
        Args:
            image_path (str): Path to the receipt image
            preprocess (bool): Whether to preprocess the image
            
        Returns:
            dict: Dictionary with all extracted information
        """
        print(f"Processing receipt: {image_path}")
        
        # Extract text from image
        text_results, full_text = self.extract_text(image_path, preprocess)
        
        if not text_results:
            print("No text detected in the image")
            return {
                'success': False,
                'error': 'No text detected in the image'
            }
        
        print(f"Extracted {len(text_results)} text elements")
        
        # Identify receipt type
        receipt_type = self.identify_receipt_type(full_text)
        print(f"Identified receipt type: {receipt_type}")
        
        # Extract merchant name
        merchant_name = self.extract_merchant_name(text_results, receipt_type)
        
        # Extract date and time
        date, time = self.extract_date_time(full_text)
        
        # Extract items
        items = self.extract_items(text_results, receipt_type, full_text)
        
        # Extract totals
        totals = self.extract_totals(full_text)
        
        # Calculate or verify total
        total = self.calculate_total(items, totals)
        
        # Extract payment method
        payment_method = self.extract_payment_method(full_text)
        
        # Prepare result
        result = {
            'success': True,
            'merchant_name': merchant_name,
            'receipt_type': receipt_type,
            'date': date,
            'time': time,
            'items': items,
            'subtotal': totals['subtotal'],
            'tax': totals['tax'],
            'service_charge': totals['service_charge'],
            'total': total,
            'payment_method': payment_method,
            'raw_text': full_text
        }
        
        return result
    
    def save_result(self, result, output_path):
        """
        Save processing result to a JSON file
        
        Args:
            result (dict): Processing result
            output_path (str): Path to save the result
            
        Returns:
            str: Path to the saved file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Result saved to {output_path}")
        return output_path
    
    def generate_summary(self, result, output_path=None):
        """
        Generate a human-readable summary of the processing result
        
        Args:
            result (dict): Processing result
            output_path (str, optional): Path to save the summary
            
        Returns:
            str: Summary text
        """
        if not result['success']:
            return f"Error: {result.get('error', 'Unknown error')}"
        
        lines = []
        lines.append('=== RECEIPT SUMMARY ===')
        lines.append(f"Merchant: {result['merchant_name']}")
        
        # Format date and time consistently
        if result['date'] or result['time']:
            date_time = f"{result['date']} {result['time']}".strip()
            lines.append(f"Date/Time: {date_time}")
        
        lines.append('')
        lines.append('Items:')
        
        if result['items']:
            for item in result['items']:
                name = item['name']
                quantity = item['quantity']
                price = item['price']
                category = item['category']
                
                lines.append(f"- {quantity}x {name} ({category}): Rp{price:,.0f}")
        else:
            lines.append("- No items detected")
        
        lines.append('')
        
        if result['subtotal']:
            lines.append(f"Subtotal: Rp{result['subtotal']:,.0f}")
        
        if result['tax']:
            lines.append(f"Tax: Rp{result['tax']:,.0f}")
        
        if result['service_charge']:
            lines.append(f"Service Charge: Rp{result['service_charge']:,.0f}")
        
        # Always include total
        lines.append(f"Total: Rp{result['total']:,.0f}")
        
        lines.append(f"Payment Method: {result['payment_method']}")
        
        summary = '\n'.join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"Summary saved to {output_path}")
        
        return summary
