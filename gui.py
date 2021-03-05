import sys
from PyQt5 import QtCore, QtWidgets
from sibiris import Sibiris
from PyQt5.QtGui import QIcon, QPixmap

class Analysis(QtWidgets.QWidget):

    switch_window = QtCore.pyqtSignal(list)

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle('SIBIRIS - analysis')

        self.layout = QtWidgets.QGridLayout()
        self.sibiris = Sibiris()
        self.files = []
        self.analysing = False
        
        self.button2 = QtWidgets.QPushButton('Import Images')
        self.button2.clicked.connect(self.importImages)
        self.layout.addWidget(self.button2)
        
        self.label2 = QtWidgets.QLabel(self)
        file_names = [f.split("/")[-1] for f in self.files]
        self.label2.setText("Images to be analysed: " + str(file_names))
        self.label2.setWordWrap(True)
        self.layout.addWidget(self.label2)
        self.button = QtWidgets.QPushButton('Start Processing')
        self.button.clicked.connect(self.processImages)
        self.layout.addWidget(self.button)
        self.label = QtWidgets.QLabel(self)
        self.label.setText('Please wait while the images are processed. This may take a while.')
        self.layout.addWidget(self.label)

        self.setLayout(self.layout)

    def processImages(self):
        if not self.analysing and self.files:
            print("Analyse!")
            self.analysing = True
            results = self.sibiris.processImages(self.files)
            print(results)
            self.switch_window.emit(results)
            
    def importImages(self):
        options = QtWidgets.QFileDialog.Options()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Python Files (*.py)", options=options)
        if files:
            self.files = files
            file_names = [f.split("/")[-1] for f in self.files]
            self.label2.setText("Images to be analysed: " + str(file_names))
            self.resize(self.label2.height(),self.label2.width())
        self.update()
        
        

class Results(QtWidgets.QWidget):
    
    switch_window = QtCore.pyqtSignal(list)
    
    def __init__(self,results,index):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle('SIBIRIS - results')
        self.results = results
        layout = QtWidgets.QGridLayout()
        groupBox_0 = QtWidgets.QGroupBox()
        layout_0 = QtWidgets.QGridLayout()
        label1 = QtWidgets.QLabel(self)
        pixmap = QPixmap(results[index][0])
        label1.setPixmap(pixmap.scaled(500,500,  QtCore.Qt.KeepAspectRatio))
        #self.resize(pixmap.width(),pixmap.height())
        layout_0.addWidget(label1)
        self.label = QtWidgets.QLabel('Input Image: ' + str(results[index][0]))
        layout_0.addWidget(self.label)
        groupBox_0.setLayout(layout_0)
        layout.addWidget(groupBox_0)
        groupBox = QtWidgets.QGroupBox()
        layout_gb = QtWidgets.QGridLayout()#QtWidgets.QHBoxLayout()
        
        #groupBox_r = QtWidgets.QGroupBox()
        #layout_r = QtWidgets.QGridLayout()
        self.currently_selected = None
        for i in range(1,len(results[index])):
                label = QtWidgets.QLabel(self)
                pixmap = QPixmap(results[index][i][0])
                label.setPixmap(pixmap.scaled(400,400,  QtCore.Qt.KeepAspectRatio))
                layout_gb.addWidget(label,0,i-1)
                label = QtWidgets.QLabel("Image title: " + str(results[0][i][0]))
                layout_gb.addWidget(label,1,i-1)
                label = QtWidgets.QLabel("Distance: " + str(results[0][i][1]))
                layout_gb.addWidget(label,2,i-1)
                radioButton = QtWidgets.QRadioButton("This is a match.")
                radioButton.imagetitle=results[index][i][0]
                radioButton.toggled.connect(self.onClicked)
                layout_gb.addWidget(radioButton,3,i-1)
        groupBox.setLayout(layout_gb)
        layout.addWidget(groupBox)
        self.confirm = QtWidgets.QPushButton('Confirm match')
        self.confirm.clicked.connect(self.onConfirmed)
        layout.addWidget(self.confirm)
        self.textbox = QtWidgets.QLineEdit(self)
        layout.addWidget(self.textbox)
        self.newwhale = QtWidgets.QPushButton('This is a new whale (input name above)')
        self.newwhale.clicked.connect(self.onNewWhale)
        layout.addWidget(self.newwhale)
        
        self.setLayout(layout)

    def onClicked(self):
        radioButton = self.sender()
        self.currently_selected = radioButton.imagetitle
        if radioButton.isChecked():
            print("Selected Image " + str(self.currently_selected))
            
    def onConfirmed(self):
        if self.currently_selected:
            print("Confirmed match")
            #send confrimation back to system
            #open next result
            self.switch_window.emit(self.results)
        else:
            print("please choose match first")
            #potential error message
    
    def onNewWhale(self):
        if self.textbox.text():
            print(self.textbox.text())
            
            #send new whale name and tag to system and confirm that name is unique
            #open next result
            self.switch_window.emit(self.results)
        else:
            print("please input name first")
        

class Controller:

    def __init__(self):
        self.results_index = 0
        pass

    

    def show_analysis(self):
        self.analysis = Analysis()
        self.analysis.switch_window.connect(self.show_results)
        self.analysis.show()
        

    def show_results(self,results):
        if self.results_index < len(results):
            if self.results_index == 0:
                self.analysis.close()
            else:
                self.results.close()
            self.results = Results(results,self.results_index)
            self.results.switch_window.connect(self.show_results)
            self.results_index += 1
            self.results.show()
        else:
            self.results.close()


def main():
    app = QtWidgets.QApplication(sys.argv)
    controller = Controller()
    controller.show_analysis()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()