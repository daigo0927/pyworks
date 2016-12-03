#include <iostream>
#include <fstream>
#include <string>
using namespace std;
int main(){
    ifstream ifs("train.txt");
    string filename;
    int num;
    cout << "<?xml version='1.0' encoding='ISO-8859-1'?>" << endl
        << "<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>" << endl
        << "<dataset>" << endl
        << "<name>imglab dataset</name>" << endl
        << "<images>" << endl;
    while(ifs >> filename >> num && num){
        cout << "<image file=\'" << filename << "\'>" << endl;
        for(int i=0; i<num ; ++i){
           int x, y, width, height;
            ifs >> x >> y >> width >> height;
            cout << "<box top=\'" << y << "\' left=\'" << x
                << "\' width=\'" << width << "\' height=\'" << height << "\'/>" << endl;
        }
        cout << "</image>" << endl;
    }
    cout << "</images>" << endl;
    cout << "</dataset>" << endl;
    return 0;
}