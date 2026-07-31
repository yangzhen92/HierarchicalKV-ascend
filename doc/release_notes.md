# 版本说明

## 版本配套说明<a name="ZH-CN_TOPIC_0000002545204925"></a>

### 产品版本信息<a name="ZH-CN_TOPIC_0000002513525038"></a>

<a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108__Ref249955742"></a>
<table><tbody><tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row244mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><span id="ph925512229126"><a name="ph925512229126"></a><a name="ph925512229126"></a>HierarchicalKV-ascend</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row255mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a>26.1.0</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row7259721105019"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a>Release版本</p>
</td>
</tr>
</tbody>
</table>

### 相关产品版本配套说明<a name="ZH-CN_TOPIC_0000002513684954"></a>

**表 1**  HierarchicalKV-ascend软件版本配套表

| HierarchicalKV-ascend | CANN版本 | Ascend HDK版本      |
|-----------|--------|-------------------|
| 26.1.0    | 9.1.0  | 25.6.RC1 |

## 版本兼容性说明<a name="ZH-CN_TOPIC_0000002545284919"></a>

- HierarchicalKV-ascend：本版本无兼容性问题。

**表 2**  HierarchicalKV-ascend与CANN版本兼容

<table style="table-layout: fixed; width: 433px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">HierarchicalKV-ascend</th>
    <th colspan="1">CANN版本</th>
  </tr>
  <tr>
    <th>9.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>26.1.0</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**表 3**  HierarchicalKV-ascend与Ascend HDK版本兼容

<table style="table-layout: fixed; width: 433px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">HierarchicalKV-ascend</th>
    <th colspan="1">Ascend HDK版本</th>
  </tr>
  <tr>
    <th>25.6.RC1</th>
  </tr></thead>
<tbody>
  <tr>
    <td>26.1.0</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

## 版本使用注意事项<a name="ZH-CN_TOPIC_0000002513684952"></a>

无

## 更新说明<a name="ZH-CN_TOPIC_0000002545284923"></a>

### 新增特性<a name="ZH-CN_TOPIC_0000002545284925"></a>

| 特性描述 | 配套产品型号 |
|--------|------------|
| 支持HBM+DDR分级存储模式 | Atlas 950系列 |
| 支持全量社区接口 | Atlas 950系列 |
| 支持全量淘汰策略 | Atlas 950系列 |
| 新增is_pure_hbm_mode接口 | Atlas 950系列 |

### 业务接口变更<a name="ZH-CN_TOPIC_0000002545204929"></a>

新组件不涉及接口变更。

### 关键特性变更<a name="ZH-CN_TOPIC_0000002513684958"></a>

新组件不涉及关键特性变更。

### 已解决的问题<a name="ZH-CN_TOPIC_0000002513525034"></a>

无

### 遗留问题<a name="ZH-CN_TOPIC_0000002513525040"></a>

无

## 升级影响<a name="ZH-CN_TOPIC_0000002545284921"></a>

### 升级过程对现行系统的影响<a name="ZH-CN_TOPIC_0000002545204927"></a>

无

### 升级后对现行系统的影响<a name="ZH-CN_TOPIC_0000002513525036"></a>

无

## 26.1.0版本配套文档<a name="ZH-CN_TOPIC_0000002545204931"></a>

| 文档名称                                          | 内容简介                                         | 更新说明                                               |
|-----------------------------------------------|----------------------------------------------|----------------------------------------------------|
| 《[HierarchicalKV-ascend 26.1.0 用户指南](../README.md)》 | 主要包括HierarchicalKV-ascend的简介、概述、关键特性、安装部署、快速使用、API参考等内容。 | 变更详见《[HierarchicalKV-ascend 26.1.0 用户指南](../README.md)》。 |

## 病毒扫描结果

病毒扫描通过。

## 漏洞修补列表<a name="ZH-CN_TOPIC_0000002545204923"></a>

无
