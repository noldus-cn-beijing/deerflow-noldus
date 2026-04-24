# Case-001 Shoaling Baseline 分析笔记

> 工程侧初稿，行为学同事待补。标注 TODO(行为学同事) 的部分需要专家判断。

## 1. 数据背景

5 条成年斑马鱼的群体行为轨迹数据，由 EthoVision XT 180 采集。分为 2 组：
- **control**: Subject 1, Subject 2 (n=2)
- **treatment**: Subject 3, Subject 4, Subject 5 (n=3)

注意：这是演示数据，分组不代表真实实验处理。数据用于验证 agent 分析流程的端到端正确性。

TODO(行为学同事): 如果这是真实实验，control/treatment 分别对应什么处理？测试时长多少分钟？水温、光照等环境条件？

ANSWER：`control/treatment` 只是 `对照组/实验组` 的区别，在导出的项目中，5条斑马鱼并未进行实际分组。`实验项目` - `实验设置` 中的描述如下
  ```text
  In this experiment, five unmarked adult zebrafish were tracked with the aim of quantifying their shoaling behavior. Video courtesy of Robert Gerlai, Department of Psychology, University of Toronto Mississauga, Mississauga, ON, Canada.
  ```
一般来说，`control` 和 `treatment`可能是 病理/毒理/转基因造模，年龄，剂量等不同，需要在分析前由用户提供信息。没有提供的话需要对话核实。

同理，`测试时长`、`光照` 等信息也由用户提供。

## 2. 初看数据的第一印象

从指标表可以立即看到：

- **Subject 3 的 mean_nnd = 70.02 mm**，远高于其他 4 条鱼（36-40 mm 范围），约为群体均值的 1.8 倍
- **Subject 3 的 distance_moved = 12518 mm**，远低于其他个体（约 24000-26000 mm），约为群体均值的一半
- **Subject 3 的 velocity_mean = 41.8 mm/s**，远低于其余个体（约 79-87 mm/s）

这三个指标一致指向同一个体——Subject 3 的运动量偏低且离群距离偏大。

control 组 n=2 是一个明显的样本量问题，任何统计推断都需要标注此限制。

## 3. 逐个指标的判断过程

### 3.1 mean_nnd (平均最近邻距离)

正常范围 TODO(行为学同事): 斑马鱼 shoaling 实验中 mean_nnd 的典型范围是多少？受什么因素影响（鱼龄、水温、鱼缸大小）？

ANSWER：由于行为学实验受动物类型、品系、造模类型及造模程度、实验时间等诸多因素影响，一般不给定范式baseline或常模。直接以同项目实验组间对比，寻找统计检验阳性/有显著差异的结果。

当前数据：
- Subject 1: 36.09 mm, Subject 2: 39.86 mm, Subject 4: 36.36 mm, Subject 5: 38.10 mm — 这 4 条鱼聚集在 36-40 mm 的窄区间内
- Subject 3: 70.02 mm — 是群体均值(约 44 mm)的 1.6 倍

组均值：control 37.97 mm, treatment 48.16 mm。treatment 组的高值完全由 Subject 3 拉高。排除后 treatment 降至约 37.23 mm，两组接近。

### 3.2 distance_moved (总运动距离)

Subject 3 的 12518 mm 仅为其他个体（23715-26179 mm）的约 50%。如果这是真实实验，低运动量可能提示：运动能力下降、探索行为减少、或者该个体在角落停留时间过长。

TODO(行为学同事): 低于群体均值 50% 的运动量，在判断离群时应该用什么阈值？是绝对值还是相对群体均值/中位数的倍数？

ANSWER：**离群阈值不应该用总运动距离判断**。在原项目中，Shoaling behavior由下列几种自定义参数进行衡量。

- Interindividual Distance (IID)
  - 使用JavaScript Continuous 变量。
  - 计算平均值和标准差  
  内容如下：
  ```JavaScript
  //InterIndividual Distance (IID) - Continuous.js

  //Calculates the average distance of each subject to all the others and stores a value for each sample
  //Assumptions: more than one subject, center point based distance
  // Unit is mm
  
  //Enter here the names of all the subjects as they are defined in EthoVision XT

  const g_aSubjects = ["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5"];
  
  // Function to calculate distance between two points
  function Distance(pt1, pt2)
  {
    var dx = pt1.x - pt2.x;
    var dy = pt1.y - pt2.y;

    return Math.sqrt(dx * dx + dy * dy);
    }
    
  function Start()
  {

  }
  
  function Stop()
  {

  }
  
  function Process()
  {
    var ptFocal = GetCenter();
    var avg;

    if (ptFocal != null)
    {
        var x1 = ptFocal.x;
        var y1 = ptFocal.y;

        // In this iteration each subject i is considered and the distance between ptFocal and ptSubj for all Subjects i is calculated

        var i;
        var avg = 0;
        var nTotal = 0;

        for (i = 0; i < g_aSubjects.length; ++i)
        {
           var ptSubj = GetSubjectCenter(g_aSubjects[i]);

           // If the subject i exists, take the coordinates of the center 

           if (ptSubj != null)
           {
            
              var x2 = ptSubj.x;
              var y2 = ptSubj.y;

              // Check that subject i is not the same as the focal otherwise skip
 
              if (x1 != x2 && y1 != y2)
              {
                  var Dist = Distance(ptFocal, ptSubj);
                  
                  avg += Dist;
                  ++nTotal;

              }

          }

       // This is the end of the for loop
       }

      // Calculates the average distance

       if (nTotal != 0)
       {
           avg /= nTotal;
       }
       else
       {
          avg = null;
       }

    }

    // Returns the value of average distance between Focal subject and other subjects for this sample

    if (avg != null)
    {
        SetOutput(avg);

    }
    else
    {
        SetOutputMissing();
    };
    }
  ```
  - Number of subject in 4 quadrants
    - 整个水池被分为四个象限，使用4个JS Continuous自定义变量，输出四个象限随时间变化，象限内的斑马鱼条数。
    - 计算`最小值`，`平均值`和`最大值`。
    下面以第一象限为例：
    ``` JavaScript
    const g_aSubjects = ["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5", "Subject 6", "Subject 7", "Subject 8", "Subject 9", "Subject 10", "Subject 11", "Subject 12", "Subject 13", "Subject 14", "Subject 15", "Subject 16"];
    const g_Zone      = "Zone 1";
    
    function InZ(pt1)
    {
        var z = 0;
        
        if (pt1 !== null)
        {
            if (IsInZone(g_Zone, pt1))
            {
                z = 1;
            }
        }
        
        return z;
    }
    
    function Start()
    {

    }
    
    function Stop()
    {

    }
    
    function Process()
    {
        var n = 0;
        
        var i;
        
        for (i = 0; i < g_aSubjects.length; ++i)
        {
            var pt = GetSubjectCenter(g_aSubjects[i]);
            
            if (InZ(pt))
            {
                ++n;
            }
        } 
        
        SetOutput(n);
    }
    ```

  - Subject coordinates
    - 获取各条鱼的`XY坐标`
    - 记录`最小值`、`平均值`和`最大值`
    - 在EthoVision XT软件中没法直接导出，需要用自定义指标才能看；但是在我们使用的raw data里面，是天然有对应列的。

  - All fish in one quadrant
    - 是JS State类型的自定义变量，计算所有鱼
      - 都在第一象限时
      - 都在第二象限时
      - 都在第三象限时
      - 都在第四象限时
      - 都在任一象限时  
      
      的累计持续时间，和累计时间占比。
    - 另外，还有一个灵活自定义JS State变量，可以指定任何区域内有`k`条鱼时触发（k值可以自己指定）

  - Ratio of Subjects
    - 用于计算各条鱼在第一象限滞留比率。

  - Nearest Neighbor Distance
    - 计算相对各鱼而言，各个时间点最近的鱼是哪条？
    - 包括两个指标
      - Distance between subjects   
      EthoVision自带指标。计算各个被试动物，在各个时间点，互相之间的距离。输出平均值。
      - NND  (Nearest Neighbor Distance)  
      自定义JS Continuous指标，用于计算最近的邻鱼是哪条。输出平局值和标准差。代码如下：
      ```JavaScript
      //Nearest Neighbour Distance (NND) - Continuous.js
      
      //Calculates the nearest neighbour distance for each subject and at each sample
      //Assumptions: more than one subject, center point based distance
      
      //Enter here the names of all the subjects as they are defined in EthoVision XT
      
      const g_aSubjects = ["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5"];
      
      // Function to calculate distance between two points
      
      function Distance(pt1, pt2)
      {
        var dx = pt1.x - pt2.x;
        var dy = pt1.y - pt2.y;
        
        return Math.sqrt(dx * dx + dy * dy);
      }
      
      function Start()
      {

      }
      
      function Stop()
      {

      }
      
      function Process()
      {
        var ptFocal = GetCenter();
        var min_dist;
        
        if (ptFocal != null)
        {
            var x1 = ptFocal.x;
            var y1 = ptFocal.y;

        // In this iteration each subject is considered and the Distance function is called

        var i;

        for (i = 0; i < g_aSubjects.length; ++i)
        {
            var ptSubj = GetSubjectCenter(g_aSubjects[i]);

            if (ptSubj != null)
            {
                var x2 = ptSubj.x;
                var y2 = ptSubj.y;

                if (x1 != x2 && y1 != y2)
                {
                    var Dist = Distance(ptFocal, ptSubj);

                    if (min_dist == null)
                    {
                        min_dist = Dist
                    }

                    if (Dist < min_dist) 
                    {
                        min_dist = Dist;
                    }
                }
            }
        }
        }

      // Returns the value of minimum distance for this Focal subject and this sample
       
      if (min_dist != null)
      {
        SetOutput(min_dist);
        }
      else
      {
        SetOutputMissing();
      }
      }

      ```

### 3.3 velocity (速度)

Subject 3 的 mean velocity = 41.8 mm/s，max = 397 mm/s。注意 max velocity 与其他个体(384-436 mm/s)相当，说明 Subject 3 具备短时间爆发运动能力，但持续运动量低。这可能是"间歇性活跃但整体不活跃"的模式。

## 4. 异常识别与辨别

Subject 3 的异常最可能属于 **A: 个体表型变异**，理由：

- 多个指标（NND、距离、速度）一致偏离，指向同一行为模式（低活跃 + 高离群距离）
- max velocity 正常，排除运动能力损伤（D: 设备故障可以排除因为速度爆发正常）
- 没有轨迹跳变或异常坐标的报告，排除 D

**需要排除的解释**：
- C (统计离群): 如果只是统计极端值，不一定在多个指标上一致偏离。Subject 3 的多指标一致性更指向行为型差异
- B (混杂因素): 需要更多信息（体重、性别、时辰）来排除

TODO(行为学同事):
1. 你同意这个分类吗？还是认为更可能是其他类型？
2. 斑马鱼中是否存在文献报告的"探索型/活跃型"表型亚群？比例大约多少？
3. 如果这是真实药物实验，Subject 3 应该被排除还是保留在分析中？判断标准是什么？

ANSWER:
1. 当前实验由于没有实际分组，不同意这个分组解读。这也是“需要排除的解释-C”中出现问题的原因。  
但是对于实际实验，解读方法没有问题（velocity评估运动能力，nnd和iid评估离群）
2. 没有相关信息。但是不重要：因为这些会在分析前询问用户时得到。
3. 如果是真实实验，需要保留。一般只有造模失败/检测任务学习失败的时候，才会在进入下一个试验阶段时排除掉个体，且排除数量和原因需要在论文中进行报告。

## 5. 最终结论

**工程侧初稿**：组间 mean_nnd 和 distance_moved 差异均不显著（Mann-Whitney U, p=0.8 和 p=1.0，Bonferroni 校正后 α=0.025）。treatment 组的 mean_nnd 偏高主要由 Subject 3 的离群值驱动——排除后组均值从 48.16 降至 37.23 mm，与 control（37.97 mm）几乎一致。在当前样本量（control n=2, treatment n=3）下，不应将组间差异归因于处理效应。建议增加样本量后重新评估。

TODO(行为学同事): 以上结论的措辞是否专业？作为行为学研究员，你会怎么写这段结论？

ANSWER：比较专业。文章的Result和Discussion是分开的。Result仅汇报结果和统计检验信息，以及相关补充。解读是放在Discussion中的。

## 6. 参考文献（可选）

TODO(行为学同事): 补充与斑马鱼 shoaling NND 正常范围、表型亚群相关的参考文献。

ANSWER：无需补充。不考虑和常模环比。只进行实验内的组间对比。
