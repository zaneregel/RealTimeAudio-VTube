import sys
import requests
import json
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

def get_graph_data():
        url = "http://127.0.0.1:5000/data"
        response = requests.get(url)

        if response.status_code == 200:
            graph_data = json.loads(response.text)
            response.close()
            return graph_data
        else:
            print("HTTP Error:", response.status_code)
            response.close()
        
        return None

class GraphWindow(QtWidgets.QMainWindow):
    def __init__(self, graph_data):
        super().__init__()
        self.graph_data = []
        self.http_error = False
        self.setWindowTitle("Graph Viewer (Auto-Refreshing)")
        self.setGeometry(100, 100, 800, 600)

        # Create PyQtGraph plot widget
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        self.graphWidget.setYRange(-0.4, 1.4)
        self.graphWidget.hideAxis("left")
        self.graphWidget.hideAxis("bottom")
        self.graphWidget.setAspectLocked(False)

        # Set up a timer to refresh the graph periodically
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_graph)
        self.timer.start(8)  # Fetch new data every 8ms or ~120fps

        # Initial plot
        self.update_graph()

    def update_graph_data(self):
        url = "http://127.0.0.1:5000/data"
        response = requests.get(url)

        if response.status_code == 200:
            graph_data = json.loads(response.text)
            self.http_error = False
            response.close()
            return graph_data
        else:
            print("HTTP Error:", response.status_code)
            self.http_error = True
            response.close()
            return None


    def update_graph(self):
        graph_data = self.update_graph_data()
        if not graph_data:
            return

        # Create Plot points
        x_values = range(0, len(graph_data))
        y_values = []
        for i in graph_data:
            y_values.append(i)
            
        # Clear previous graph
        if(self.http_error != True):
           self.graphWidget.clear()

        # Plot the new graph
        pen = pg.mkPen(color='b', width=2)
        self.graphWidget.plot(x_values, y_values, pen=pen)

def main():
    
    graph_data = get_graph_data()
    if not graph_data:
        return

    # Start PyQtGraph application
    app = QtWidgets.QApplication(sys.argv)
    window = GraphWindow(graph_data)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()