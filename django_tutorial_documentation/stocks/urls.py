from django.urls import path
from . import views

urlpatterns= [

    path('<int:question_id>/vote/', views.vote, name='vote'),

    path('api/stock/<str:stock_symbol>/',views.get_stock, name='get_stock'),

	path('',views.MadinetNasrModels.index,name='index'),
	path('About',views.MadinetNasrModels.AboutPage,name='About'),
	path('Contact',views.MadinetNasrModels.ContactPage,name='Contact'),
	path('api/data1',views.MadinetNasrModels.ClosePrice,name='ClosePrice'),
	path('api/data2',views.MadinetNasrModels.RegressionModels,name='RegressionModels'),
	path('api/data3',views.MadinetNasrModels.ClosePriceSuez,name='ClosePriceSuez'),
	path('api/data4',views.MadinetNasrModels.RegressionModelsSuez,name='RegressionModelsSuez'),
	#path('api/data3',views.MadinetNasrModels.ForeCastModel,name='ForeCastModel'),
	#path('api/data4',views.MadinetNasrModels.LSTMModel,name='LSTMModel'),
	#path('api/data5',views.MadinetNasrModels.ArimaModel,name='ArimaModel'),
]