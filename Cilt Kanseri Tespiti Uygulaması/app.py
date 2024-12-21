from __future__ import division, print_function

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)


# Lezyon sınıfları
lesion_classes_dict = {
    0: (
        'MELANOSİTİK NEVÜS (Ben) - Melanositik nevüs, ciltte bulunan melanin üreten hücrelerden (melanositler) '
        'kaynaklanan benign (iyi huylu) benlerdir. Bu tür benler genellikle zararsızdırlar ve çoğu zaman tedavi '
        'gerektirmez. Ancak bazı durumlarda, özellikle benin büyümesi, renk değişimi veya şekil değişikliği gibi '
        'belirtiler varsa, izlenmesi ve dermatolojik değerlendirilmesi gerekebilir. Melanositik nevüsler genellikle '
        'doğal olarak doğar veya yaşamın erken dönemlerinde gelişir. Ailede melanositik nevüs öyküsü olan bireyler, '
        'daha fazla dikkat etmelidir.'
    ),
    
    1: (
        'MELANOM/KANSERLİ - Melanom, melanositlerden kaynaklanan en agresif ve tehlikeli cilt kanseri türüdür. '
        'Erken teşhis ve tedavi, melanomun başarılı bir şekilde yönetilmesinde hayati önem taşır. Melanom, derideki '
        'melanin üreten hücrelerden gelişir ve hızlı bir şekilde vücuda yayılabilir. Bu kanser türü genellikle bir '
        'benin veya mevcut bir nevüsün aniden büyümesi, renginin değişmesi, sınırlarının düzensizleşmesi ile '
        'kendini gösterir. Erken evrelerde tedavi edilebilirken, geç dönemde tedavi şansı düşer. Güneşe aşırı '
        'maruz kalanlar ve ailesinde melanom öyküsü bulunan kişilerde risk daha yüksektir.'
    ),
    
    2: (
        'BENİNG KERATOZ BENZERİ LEZYONLAR - Benign keratoz benzeri lezyonlar, ciltteki keratin hücrelerinin aşırı '
        'büyümesi sonucu oluşur. Bu lezyonlar genellikle zararsızdır ve çoğu zaman tedavi gerektirmez. Ancak, bazı '
        'durumlarda bu lezyonlar cilt kanserine dönüşebilir, bu yüzden takip edilmesi önemlidir. Keratoz benzeri '
        'lezyonlar genellikle sarımsı, kahverengi veya kırmızımsı renklerde olabilir ve genellikle cilt yüzeyine '
        'yükselirler. Güneş ışığına maruz kalan bölgelerde daha sık görülür ve yaşlı bireylerde yaygın olarak ortaya '
        'çıkabilir.'
    ),
    
    3: (
        'BAZAL HÜCRELİ KARSİNOM (BCC) - Bazal hücreli karsinom, cildin en üst tabakasında bulunan bazal hücrelerden '
        'kaynaklanan bir kanser türüdür. Genellikle yavaş büyür ve diğer cilt kanserlerine göre daha az agresiftir. '
        'Bazal hücreli karsinomlar genellikle güneşe maruz kalan bölgelerde, özellikle yüz, kulaklar, boyun ve ellerde '
        'görülür. Bu tür kanserler çoğu zaman zararsız gibi görünse de, tedavi edilmezse çevre dokulara yayılabilir ve '
        'büyük yara izlerine yol açabilir. Cerrahi müdahale genellikle tedavi için yeterlidir.'
    ),
    
    4: (
        'AKTİNİK KERATOZLAR - Aktinik keratoz, ciltteki ince, kabuklu ve genellikle pembe veya kırmızımsı lezyonlardır. '
        'Güneş ışığına uzun süre maruz kalan bölgelerde ortaya çıkarlar ve genellikle ciltteki keratin hücrelerinin '
        'aşırı büyümesi sonucu gelişir. Aktinik keratozlar, cilt kanseri riskini artırabilecek premalign (kötü huylu '
        'olmaya yatkın) lezyonlar olarak kabul edilir. Tedavi edilmezse, bu lezyonlar bazal hücreli karsinom veya skuamöz '
        'hücreli karsinoma dönüşebilir. Güneşe karşı duyarlı olan kişilerde daha sık görülür.'
    ),
    
    5: (
        'DAMAR LEZYONLARI - Damar lezyonları, damarlarla ilgili anormal oluşumları veya lezyonları ifade eder. Bu '
        'lezyonlar genellikle ciltte anormal kan damarları veya damar genişlemeleri şeklinde ortaya çıkarlar. Damar '
        'lezyonları, genellikle zararsızdır ve tedavi gerektirmezler. Ancak bazı damar lezyonları, çevresel faktörler, '
        'genetik faktörler veya ciltteki kan dolaşımı bozuklukları nedeniyle daha belirgin hale gelebilir. Bu tür lezyonlar '
        'genellikle kırmızı, mor veya mavi renkte olabilir ve çoğunlukla yüz, bacaklar veya vücudun diğer bölgelerinde '
        'görülür.'
    ),
    
    6: (
        'DERMATOFİBROM - Dermatofibrom, genellikle ciltte görülen, benign (iyi huylu) bir tümördür. Bu lezyonlar genellikle '
        'sert, kabarık ve kahverengi renkte olur. Dermatofibromlar genellikle ağrısızdır ve ciltte tek başlarına veya '
        'birkaç tane halinde ortaya çıkabilirler. Çoğunlukla bacaklarda, kollarda veya vücutta başka yerlerde '
        'bulunurlar. Genellikle travma veya ciltteki küçük yaralanmalar sonrası gelişirler. Tedavi gerektirmezler ancak '
        'estetik kaygılar nedeniyle cerrahi olarak çıkarılabilirler.'
    )
}


# Modeli yükle ve optimizer ekle
model = load_model("model12345.h5", compile=False)
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

@app.route('/')
def index():
    return render_template('index.html')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Model ile tahmin yap
        preds = model_predict(file_path, model)
        pred_class = preds.argmax(axis=-1)
        pr = lesion_classes_dict[pred_class[0]]
        return jsonify({'result': pr})  # JSON formatında doğru sonucu döndür

    return jsonify({'error': 'Invalid request'})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
