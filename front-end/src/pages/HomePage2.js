import React from 'react';
import ReactDOM from 'react-dom';
import { Layout, Menu, Breadcrumb, Icon, Collapse} from "antd";
import { Input, Button, message} from "antd";
import { PageHeader,  Tag, Typography,Tooltip, Popover} from 'antd';
import ApiUtil from '../Utils/ApiUtil';
import HttpUtil from '../Utils/HttpUtil';
import './HomePage2.css';

import { Row, Col } from "antd";
const { TextArea } = Input;
const { Panel } = Collapse;
const { Paragraph, Title } = Typography;
const { Header, Content, Footer, Sider } = Layout;
const { SubMenu } = Menu;
var _re;
const customPanelStyle = {
  background: "#f7f7f7",
  borderRadius: 4,
  marginBottom: 24,
  border: 0,
  overflow: 'hidden',
};

const IconLink = ({ src, text }) => (
  <a
    style={{
      marginRight: 16,
      display: 'flex',
      alignItems: 'center',
    }}
  >
    <img
      style={{
        marginRight: 8,
      }}
      src={src}
      alt="start"
    />
    {text}
  </a>
);
const content = (
  <div className="content">
    
    <Row className="contentLink" type="flex">
      <a href = "https://www.baidu.com">
      <IconLink
        src="https://gw.alipayobjects.com/zos/rmsportal/MjEImQtenlyueSmVEfUD.svg"
        text="Introduction"
      
      />
      </a>
      
      <IconLink
        src="https://gw.alipayobjects.com/zos/rmsportal/ohOEPSYdDTNnyMbGuyLb.svg"
        text="Structure"
      />

      <IconLink
        src="https://gw.alipayobjects.com/zos/rmsportal/NbuDUAuBlIApFuDvWiND.svg"
        text="Q & A"
      />
    </Row>
    <br/>
    <Paragraph>
      Based on fine tuned XLNet
    </Paragraph>
    <Paragraph>
      Support classification and generation tasks
    </Paragraph>
  </div>
);
const content_1 = (
  <div>
    <img src="./1.jpg" />
  </div>
);


const Content_H = ({ children, extraContent }) => {
  return (
    <Row className="content" type="flex">
      <div className="main" style={{ flex: 1 }}>
        {children}
        
      </div>
      <div
        className="extra"
        style={{
          marginLeft: 80,
        }}
      >
        {extraContent}
      </div>
    </Row>
  );
};
class HomePage extends React.Component {
  state = {
    collapsed: true,
    msg:'',
    inputdata:'',
    text_g:'',
    text_c:'',
    flag:0,
  };

  

  onCollapse = collapsed => {
    console.log(collapsed);
    this.setState({ collapsed });
  };
  updateState=(e)=>{
    this.setState({inputdata:e.target.value})
  }
  clearInput=()=>{
    this.setState({inputdata:''})
    this.myInput.focus()
    this.setState({
      flag: 0
    });
  }
  saveDataCassifier=()=>{
    let text_ = this.text_;
    console.log('upload');
    
    var _r = document.getElementById("data").value;
    let _result = {
      'type':"classfication",
      'data':_r,
      }
    
    console.log((_result));
    console.log(typeof(_result));
    HttpUtil.post(ApiUtil.API_STAFF_UPDATE, _result)
      .then(
        re=>{
          console.log("返回结果：",re);
          // console.log("返回结果：",re.result);
          // console.log("返回结果：",typeof(re.message));
          // message.info(re.message);
          message.info("success");
          _re = re.result;
          // console.log("sss",_re);
          this.setState({
            msg: _re
          });
          this.setState({
            text_c: _re
          });
          this.setState({
            flag: 1
          });
          
        }
      )
      .catch(error=>{
        console.log("error");
        message.error(error.message);
      });
  };
  saveDataGenerate=()=>{
    let text_ = this.text_;
    console.log('upload');
    
    var _r = document.getElementById("data").value;
    let _result = {
      'type':"generation",
      'data':_r,
      }
    
    console.log((_result));
    console.log(typeof(_result));
    HttpUtil.post(ApiUtil.API_STAFF_UPDATE, _result)
      .then(
        re=>{
          console.log("返回结果：",re);
          // console.log("返回结果：",re.result);
          // console.log("返回结果：",typeof(re.message));
          //message.info(re.message);
          message.info("success");
          _re = re.result;
          // console.log("sss",_re);
          this.setState({
            msg: _re
          });
          this.setState({
            text_g: _re
          });
          this.setState({
            flag: 2
          });
        }
      )
      .catch(error=>{
        console.log("error");
        message.error(error.message);
      });
  }
  render() {
    return (
      <Layout style={{ minHeight: "100vh" }}>
        <Sider
          collapsible
          collapsed={this.state.collapsed}
          onCollapse={this.onCollapse}
        >
          <div className="logo" />
          <Menu theme="dark" defaultSelectedKeys={["1"]} mode="inline">
            <SubMenu
              key="sub1"
              title={
                <span>
                  <Icon type="user" />
                  <span>User</span>
                </span>
              }
            >
              <Menu.Item key="3">Team08</Menu.Item>
              
            </SubMenu>

            <Menu.Item key="9">
              <Icon type="file" />
              <span>File</span>
            </Menu.Item>
          </Menu>
        </Sider>

        <Layout>
          
          <PageHeader
    title="TEAM08"
    style={{
      border: "1px solid rgb(235, 237, 240)"
    }}
    subTitle="task 01"
    tags={<Tag color="orange">Running</Tag>}
    
    extra={[
      
      <a href="https://github.com/">
        <Button key="1" type="primary" >
          Github
        </Button>
      </a>,
      <Popover content={content_1} >
        <Button key="2" type="primary" >
          Reference
        </Button>
      </Popover>,
      
    ]}
    // team 08前面的图片
    avatar={{
      src: "https://avatars1.githubusercontent.com/u/8186664?s=460&v=4"
    }}
    // breadcrumb={{ routes }}
  >
    <Content_H
      extraContent={
        <img
          src="https://gw.alipayobjects.com/mdn/mpaas_user/afts/img/A*KsfVQbuLRlYAAAAAAAAAAABjAQAAAQ/original"
          alt="content"
        />
      }
    >
      {content}
    </Content_H>
  </PageHeader>
          <Content style={{ margin: "0 16px" }}>
            {/* <Breadcrumb style={{ margin: "16px 0" }}>
              <Breadcrumb.Item>User</Breadcrumb.Item>
              <Breadcrumb.Item>3</Breadcrumb.Item>
            </Breadcrumb> */}
            <div style={{ margin: "24px 0" }} />
            <h3>提交文本</h3>
            <div style={{ margin: "24px 0" }} />
            <TextArea
              placeholder="Autosize height "
              autosize={{ minRows: 5, maxRows: 10 }}
              ref={input => this.text_ = input}
              id='data'
              value={this.state.inputdata} 
              onChange={this.updateState} 
              ref={myInput=>this.myInput=myInput}
            />
            <div style={{ margin: "24px 0" }} />
            <Row type="flex" justify="space-between" align="middle">
              <Col span={3}>
                <Tooltip placement="topLeft" title="点击进行分类" arrowPointAtCenter>
                  <Button type ='primary' onClick={()=>{this.saveDataCassifier(this)}} block>分类</Button >
                </Tooltip>
              </Col>
              <Col span={3}>
                <Tooltip placement="topLeft" title="点击进行生成" arrowPointAtCenter>
                  <Button type ='danger' onClick={()=>{this.saveDataGenerate(this)}} block>生成</Button >
                </Tooltip>
              </Col>
              <Col span={3}>
                <Tooltip placement="topLeft" title="点击清空输入" arrowPointAtCenter>
                  <Button type ='dashed' onClick={this.clearInput} block>清除</Button >
                </Tooltip>
              </Col>
            </Row>
            <div style={{ margin: "100px 0" }} />
            
            <Collapse
              bordered={false}
              activeKey={this.state.flag}
              expandIcon={({ isActive }) => (
                <Icon type="caret-right" rotate={isActive ? 90 : 0} />
              
              )}
            >
              <Panel header="分类结果" key="1" style={customPanelStyle} 
                > 

                <TextArea
                  placeholder="Result"
                  autosize={{ minRows: 3, maxRows: 10 }}
                  ref={input => this.text_ = input}
                  id='data2'
                  value={this.state.text_c}
                />
              </Panel>
              <Panel header="生成结果" key="2" style={customPanelStyle}>
              <TextArea
                  placeholder="Result"
                  autosize={{ minRows: 3, maxRows: 10 }}
                  ref={input => this.text_ = input}
                  id='data2'
                  value={this.state.text_g}
                />
              </Panel>
            </Collapse>
          </Content>
          <Footer style={{ textAlign: "center" }}>
          Cloud Computing Final Project Demo ©2019 Created by team08
          </Footer>
        </Layout>
      </Layout>
    );
  }
}

export default HomePage;
