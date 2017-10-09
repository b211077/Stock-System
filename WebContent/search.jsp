<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<%@ page import="java.util.*"%>
<%@ page import="stock.model.*"%>
<%@ page import="stock.model.dto.*"%>
<%
	String json;
	String data = request.getParameter("id");
	String type = request.getParameter("type");
	
	StockService sDAO = new StockService();
	
	if(data == null){
		ArrayList<StockDTO> search = null;
		search = sDAO.searchList("0");
		json = Converter.convertToJson(search);
	} else if(data.equals("top10")){
		ArrayList<StockDTO> search = null;
		search = sDAO.searchList("1");
		json = Converter.convertToJson(search);
	} 	else{
		StockService search = null;
		search = sDAO.tourSpotInfo(data);
		System.out.println(search.getImage_url1());
		json = Converter.convertToJson(search);
	}
%>

<%=json %>
