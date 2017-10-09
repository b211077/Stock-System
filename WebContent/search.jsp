<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<%@ page import="java.util.*"%>
<%@ page import="stock.model.*"%>
<%@ page import="stock.model.dto.*"%>
<%
	String json;
	String data = request.getParameter("id");
	//String type = request.getParameter("type");
	
	StockDAO sDAO = new StockDAO();
	
	if(data == null){
		ArrayList<StockDTO> search = null;
		search = sDAO.getAllStock();
		json = Converter.convertToJson(search);
	}else{
		StockDTO search = null;
		search = sDAO.getStock(data);
		//System.out.println(search.getImage_url1());
		json = Converter.convertToJson(search);
	}
%>

<%=json %>
