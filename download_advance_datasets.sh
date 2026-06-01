# 默认: CIRCO + FashionIQ + HP-FashionIQ (不下 PinPoint)
python download_advanced_datasets.py

# 只下一个
#python download_advanced_datasets.py --only fashioniq
#python download_advanced_datasets.py --only pinpoint    # 显式要才下

# CIRCO 不下 19GB 图片 (annotations 够用先)
#python download_advanced_datasets.py --only circo --skip-coco-images

# 只下载不转换
#python download_advanced_datasets.py --no-normalize